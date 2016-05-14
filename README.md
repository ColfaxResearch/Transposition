// This code supplements the white paper
//    "Multithreaded Transposition of Square Matrices
//     with Common Code for 
//     Intel Xeon Processors and Intel Xeon Phi Coprocessors"
// available at the following URL:
//     http://colfaxresearch.com/multithreaded-transposition-of-square-matrices-with-common-code-for-intel-xeon-processors-and-intel-xeon-phi-coprocessors/
// You are free to use, modify and distribute this code as long as you acknowledge
// the above mentioned publication.
// (c) Colfax International, 2013

To run the benchmark, execute the script ./benchmark.sh

The included Makefile and script ./benchmark.sh are designed for Linux.

In order for the CPU code to compile, you must have the
Intel C++ compiler installed in the system.

In order to compile and run the MIC platform code,
you must have an Intel Xeon Phi coprocessor in the system 
and the MIC Platform Software Stack (MPSS) installed and running 