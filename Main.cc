// This code supplements the white paper
//    "Multithreaded Transposition of Square Matrices
//     with Common Code for 
//     Intel Xeon Processors and Intel Xeon Phi Coprocessors"
// available at the following URL:
//     http://research.colfaxinternational.com/post/2013/08/12/Trans-7110.aspx
// You are free to use, modify and distribute this code as long as you acknowledge
// the above mentioned publication.
// (c) Colfax International, 2013

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "Transpose.h"

void InitMatrix(FTYPE* A, const int n) {
  // Initialize the matrix with a pre-determined pattern
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      A[i*n+j] = (FTYPE)(i*n+j);
}

FTYPE VerifyTransposed(FTYPE* A, const int n) {
  // Calculate the norm of the deviation of the transposed matrix elements
  // from the pre-determined pattern
  FTYPE err = 0;
#pragma omp parallel for reduction(+:err)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      const FTYPE diff = (A[i*n+j] - (FTYPE)(j*n+i));
      err += diff*diff;
    }
  return sqrt(err);
}

void FlushCaches(char* const cacheFlush, const int nCacheFlush) {
  // To avoid the retention of small matrices in cache, read/write a large dummy array
#pragma omp parallel for schedule(guided,1)
  for (int i = 0; i < nCacheFlush; i += 64)
    cacheFlush[i] = (cacheFlush[i] + cacheFlush[i+1])/2;
}

int main(){

  const int nPlans = 2;     // Number of tile traversal plans
  const int nTrials = 22;   // Number of transposition trials to measure the performance
  const int skipTrials = 2; // Skip the first two trials (initialization, verification in trial 0)
  const int nSizes = 21;    // How many matrix sizes to test

  // These are the tested matrix sizes.
  // The first size is a "good" size, which produces the best performance.
  // The second size is a "bad one", in which the matrix row does not pack an integer number of cache lines
  // The third one is an "ugly" size, for which the stride of data access is a multiple of the cache set conflict
  // In practice, always use "good" sizes; pad the matrix dimensions data to a "good" value if necessary.
#ifdef DOUBLE
  const int size[nSizes] = { 528,  524,  512,
			     1040, 1030, 1024,
			     2064, 2060, 2048,
			     4192, 4100, 4096,
			     8240, 8210, 8192,
			     16400, 16390, 16384,
			     22000,22004,21504 };
#else
  const int size[nSizes] = { 1040, 1030, 1024,
			     2064, 2060, 2048,
			     4192, 4100, 4096,
			     8240, 8210, 8192,
			     16400, 16390, 16384,
			     22000, 22004, 21504,
			     31200, 31100, 30720 };
#endif

  for (int s = 0; s < nSizes; s++) {
    
    // Matrix size to test
    const int n = size[s];

    // Matrix data container, aligned on a 64-byte boundary
    FTYPE* A = (FTYPE*)_mm_malloc(n*n*sizeof(FTYPE), 64);

    const size_t nCacheFlush = ((size_t)n*(size_t)n*sizeof(FTYPE) < (1L<<27L) ? 1L << 27L : 1L);
    char* cacheFlush = (char*)malloc(nCacheFlush);

    for (int iPlan = 0; iPlan < nPlans; iPlan++) {

      // iPlan=0 - recursive tile traversal algorithm
      // iPlan=1 - nested tile traversal algorithm

      // Plan the tile traversal order ahead of time
      int* plan = NULL;
      CreateTranspositionPlan(iPlan, plan, n);

      // Container for benchmark results
      double t[nTrials];

      for (int iTrial = 0; iTrial < nTrials; iTrial++) {

	// For the first trial only, initialize the matrix
	if (iTrial == 0) InitMatrix(A, n);

	// Perform and time the transposition operation
	const double t0 = omp_get_wtime();
	Transpose(A, n, plan);
	const double t1 = omp_get_wtime();

	if (iTrial == 0) {
	  // For the first trial only, verify the result of the transposition
	  if (VerifyTransposed(A,n) > 1e-6) {
	    fprintf(stderr,"Result of transposition is incorrect!\n");
	    exit(1);
	  }
	}

	// To simulate a realistic situation where the matrix at the beginning
	// of a calculation is not in the cache, clear the previously transposed
	// matrix from caches
	FlushCaches(cacheFlush, nCacheFlush);
	
	// Record the benchmark result
	t[iTrial] = t1-t0;

      }

      // Clean up the transposition plan data
      DestroyTranspositionPlan(plan);
    
      // Calculating transposition rate statistics
      double Ravg = 0;
      double Rmin = 1e100;
      double Rmax = 0;
      double dR = 0;
      for (int i = skipTrials; i < nTrials; i++) {
	// Transposition rate in GB/s:
	const double R = sizeof(FTYPE)*(FTYPE)2*(FTYPE)n*(FTYPE)n/t[i]/(FTYPE)(1<<30);
	Ravg += R;
	dR += R*R;
	Rmin = (R < Rmin ? R : Rmin); // Minimum observed value
	Rmax = (R > Rmax ? R : Rmax); // Maximum observed value
      }
      Ravg /= (nTrials - skipTrials); // Mean
      dR = sqrt(dR/(nTrials-skipTrials) - Ravg*Ravg); // Mean square deviation

      int goodFlag = true;
      int badFlag = false;
      int uglyFlag = false;

      // Warning about the "bad" matrix sizes
      const size_t ALIGN_BYTES = 64; // SIMD alignment requirement
      if (n % (ALIGN_BYTES/sizeof(FTYPE)) != 0) {
	// Matrix dimension n is not a multiple of the ALIGN_BYTES/sizeof(FTYPE)
	// Transposition performance may be highly suppressed by misaligned gather/scatter.
	badFlag = true;
	goodFlag = false;
      }

      // Warning about the "ugly" matrix sizes
      const int STRIDE_BYTES = 4096; // "Bad" memory stride for the LRU cache policy
      const int maxMultiple = 1;
      if ((n*sizeof(FTYPE)*maxMultiple) % STRIDE_BYTES == 0) {
	// Matrix dimension n the length of (maxMultiple) rows a multiple of (STRIDE_BYTES) bytes.
	// Transposition performance may be highly suppressed by cache set conflicts.
	uglyFlag = true;
	goodFlag = false;
      }

      printf("n=%6d  plan: %10s  rate= %6.1f +- %4.1f GB/s    range=[ %6.1f ... %6.1f ]    \"%s\"\n",
	     n, (iPlan == 0 ? "nested" : (iPlan == 1 ? "recursive" : "unknown")), Ravg, dR, Rmin, Rmax, 
	     (goodFlag ? "good" : (badFlag ? "bad" : (uglyFlag ? "ugly" : "") ) ));
      fflush(0);

    }

    free(cacheFlush);

    _mm_free(A);
  }

}
