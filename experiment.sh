#!/bin/bash

cd /trireme

source venv/bin/activate

python spmv.py benchmark -r 100 -c 2000000000 -d 0.0005  --repetitions 10

for pd in {20..5000}; do

    OMP_PROC_BIND=spread OMP_PLACES=cores python spmv_runahead.py benchmark -r 100 -c 2000000000 -d 0.0005 -pd $pd --repetitions 10
    OMP_PROC_BIND=spread OMP_PLACES=cores python spmv_runahead.py benchmark -v omp-tasks-2 -r 100 -c 2000000000 -d 0.0005 -pd $pd --repetitions 10

done
