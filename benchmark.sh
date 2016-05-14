#!/bin/bash

# This code supplements the white paper
#    "Multithreaded Transposition of Square Matrices
#     with Common Code for 
#     Intel Xeon Processors and Intel Xeon Phi Coprocessors"
# available at the following URL:
#     http://research.colfaxinternational.com/post/2013/08/12/Trans-7110.aspx
# You are free to use, modify and distribute this code as long as you acknowledge
# the above mentioned publication.
# (c) Colfax International, 2013

make clean
make

# Benchmark on the host processor (CPU), choosing the maximum available number of threads
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=`cat /proc/cpuinfo | grep processor | wc -l`
echo "Benchmarking on CPU using ${OMP_NUM_THREADS} threads with KMP_AFFINITY=${KMP_AFFINITY}:"
echo "Singe precision..."
./runme-sp-CPU > results-sp-CPU.txt
echo "Double precision..."
./runme-dp-CPU > results-dp-CPU.txt

# Benchmark on the coprocessor (MIC). Copying the executable and executing via SSH.
# This is the true native execution mode, a little faster than via micnativeloadex
scp /opt/intel/composerxe/lib/mic/libiomp5.so ./runme-sp-MIC ./runme-dp-MIC mic0:~/
MIC_THREADS=`ssh mic0 "cat /proc/cpuinfo" | grep processor | wc -l`
echo "Benchmarking on MIC using ${MIC_THREADS} threads with KMP_AFFINITY=${KMP_AFFINITY}:"
echo "Singe precision..."
ssh mic0 "KMP_AFFINITY=${KMP_AFFINITY} OMP_NUM_THREADS=${MIC_THREADS} LD_LIBRARY_PATH=. ./runme-sp-MIC" > results-sp-MIC.txt
echo "Double precision..."
ssh mic0 "KMP_AFFINITY=${KMP_AFFINITY} OMP_NUM_THREADS=${MIC_THREADS} LD_LIBRARY_PATH=. ./runme-dp-MIC" > results-dp-MIC.txt
ssh mic0 "rm libiomp5.so runme-sp-MIC runme-dp-MIC"

echo "Benchmark is complete, see files results-*.txt for data"
