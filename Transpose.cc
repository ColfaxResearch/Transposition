// This code supplements the white paper
//    "Multithreaded Transposition of Square Matrices
//     with Common Code for 
//     Intel Xeon Processors and Intel Xeon Phi Coprocessors"
// available at the following URL:
//     http://research.colfaxinternational.com/post/2013/08/12/Trans-7110.aspx
// You are free to use, modify and distribute this code as long as you acknowledge
// the above mentioned publication.
// (c) Colfax International, 2013

#include "Transpose.h"
#include <cstdlib>

const int TILE = 32; // Empirically chosen tile size

void Transpose(FTYPE* const A, const int n, const int* const plan) {

  // nEven is a multiple of TILE
  const int nEven = n - n%TILE;
  const int wTiles = nEven / TILE;                  // Complete tiles in each dimens.
  const int nTilesParallel = wTiles*(wTiles - 1)/2; // # of tiles in the matrix body

#pragma omp parallel
  { 

#pragma omp for schedule(guided)
    for (int k = 0; k < nTilesParallel; k++) {

      // For large matrices, most of the work takes place in this loop.

      const int ii = plan[2*k + 0];
      const int jj = plan[2*k + 1];

      // The main tile transposition microkernel
      for (int j = jj; j < jj+TILE; j++) { 
#pragma simd
	for (int i = ii; i < ii+TILE; i++) {
	    
	  const FTYPE c = A[i*n + j];
	  A[i*n + j] = A[j*n + i];
	  A[j*n + i] = c;

	}
      }
    }

    // Transposing the tiles along the main diagonal
#pragma omp for schedule(static)
    for (int ii = 0; ii < nEven; ii += TILE) {
      const int jj = ii;

      for (int j = jj; j < jj+TILE; j++) 
#pragma simd
	for (int i = ii; i < j; i++) {
	  const FTYPE c = A[i*n + j];
	  A[i*n + j] = A[j*n + i];
	  A[j*n + i] = c;
	}
    }

    // Transposing ``peel'' around the right and bottom perimeter
#pragma omp for schedule(static)
    for (int j = 0; j < nEven; j++) {
      for (int i = nEven; i < n; i++) {
	const FTYPE c = A[i*n + j];
	A[i*n + j] = A[j*n + i];
	A[j*n + i] = c;
      }
    }

  }

  // Transposing the bottom-right corner
  for (int j = nEven; j < n; j++) {
    for (int i = nEven; i < j; i++) {
      const FTYPE c = A[i*n + j];
      A[i*n + j] = A[j*n + i];
      A[j*n + i] = c;
    }
  }

}



void CreateTranspositionPlan(const int iPlan, int* & plan, const int n) {

  // This function must be called prior to calling the function Transpose()
  // to allocate and fill the array plan[], which contains the tile traversal order.

  // Number of complete tiles in each dimension
  const int wTiles = n / TILE;

  // Number of complete tiles below the mail diagonal in the whole matrix
  const int nTilesParallel = (wTiles-1)*wTiles/2; 

  // Request to plan
  plan = (int*) malloc(sizeof(int)*(2*nTilesParallel));

  int i = 0;
  if (iPlan == 0) {

    // Recursive tile traversal algorithm. Instead of implementing
    // a true recursive algorithm, we traverse depth-first a graph in
    // which the top node is the full matrix, and the children 
    // of every node are two halves of the parent node.
    // This allows to avoid problems with running out of stack space
    // for large matrices, where recursion may be too deep.
    
    typedef struct {
      int xStart, xEnd, yStart, yEnd;
      int parentIQ;
      int nChildren;
      int nProcessedChildren;
    } SubMatrixType;

    SubMatrixType* lifoQueue;
    int iQ = 0;
    lifoQueue = (SubMatrixType*) malloc(sizeof(SubMatrixType)*nTilesParallel);
    SubMatrixType *sM = &lifoQueue[iQ];
    sM->yStart = 1;
    sM->yEnd   = wTiles;
    sM->xStart = 0;
    sM->xEnd   = sM->yEnd-1;
    sM->parentIQ = 0;
    sM->nChildren = 0;
    sM->nProcessedChildren = 0;

    while (i < nTilesParallel ) {

      SubMatrixType *sM = &lifoQueue[iQ];
	
      if ( (sM->xEnd - sM->xStart == 1) && (sM->yEnd - sM->yStart == 1) ) {

	// Process a child
	plan[2*i + 0] = sM->xStart*TILE;
	plan[2*i + 1] = sM->yStart*TILE;
	i++;
	iQ--;
	lifoQueue[sM->parentIQ].nProcessedChildren++;

      } else if (sM->nChildren == 0) {

	// Discover children (i.e., sub-matrices)
	const int parentIQ  = iQ;
	SubMatrixType  *cM;

	// If the matrix is taller than it is wide, split it vertically; 
	// otherwise, split it horizontally
	const bool splitVertically = (sM->yEnd - sM->yStart > sM->xEnd - sM->xStart);
	const int yMid = sM->yStart + (sM->yEnd-sM->yStart)/2;
	const int xMid = sM->xStart + (sM->xEnd-sM->xStart)/2;

	// First child
	cM            = &lifoQueue[++iQ];
	cM->parentIQ  = parentIQ;
	cM->nChildren = 0;
	cM->nProcessedChildren = 0;
	if (splitVertically) {
	  cM->xStart  = sM->xStart;
	  cM->yStart  = yMid;
	} else {
	  cM->xStart  = xMid;
	  cM->yStart  = (xMid < sM->yStart ? sM->yStart : xMid+1);
	}
	cM->xEnd      = sM->xEnd;
	cM->yEnd      =   sM->yEnd;
	sM->nChildren++;

	// Second child
	cM            = &lifoQueue[++iQ];
	cM->parentIQ  = parentIQ;
	cM->nChildren = 0;
	cM->nProcessedChildren= 0;
	cM->xStart    = sM->xStart;
	cM->yStart    = sM->yStart;
	if (splitVertically) {
	  cM->xEnd    = (sM->xEnd < yMid ? sM->xEnd : yMid-1);
	  cM->yEnd    = yMid;
	} else {
	  cM->xEnd    = xMid;
	  cM->yEnd    = sM->yEnd;
	}
	sM->nChildren++;

      } else if (sM->nChildren == sM->nProcessedChildren) {
	// All children processed
	lifoQueue[sM->parentIQ].nProcessedChildren++;
	iQ--;
      } else {
	// Algorithm error, should not occur
	exit(2);
      }
    }

    free(lifoQueue);

  } else if (iPlan == 1) {

      // Nested loop plan
      for (int j = 1; j < wTiles; j++)
	for (int k = 0; k < j; k++) {
	  plan[2*i + 0] = j*TILE;
	  plan[2*i + 1] = k*TILE;
	  i++;
	}

  }

  if (i != nTilesParallel) {
    // Algorithm error, should not occur
    exit (3); 
  }

}


void DestroyTranspositionPlan(int* plan) {
  // Free the memory allocated by CreateTranspositionPlan()
  free(plan);
}


