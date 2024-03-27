#!/bin/bash

ROWS=2000
COLS=150000000

cd /trireme

source venv/bin/activate

python spmv.py benchmark -r $ROWS -c $COLS -d 0.0005 --repetitions 5

for PD in {35..300}; do

    OMP_PROC_BIND=spread OMP_PLACES=cores python spmv_runahead.py benchmark -v omp-tasks -r $ROWS -c $COLS -d 0.0005 -pd $PD -l 3 --repetitions 5

    OMP_PROC_BIND=spread OMP_PLACES=cores python spmv_runahead.py benchmark -v omp-tasks-3 -r $ROWS -c $COLS -d 0.0005 -pd $PD -l 3 --repetitions 5
    OMP_PROC_BIND=spread OMP_PLACES=cores python spmv_runahead.py benchmark -v omp-tasks-3 -r $ROWS -c $COLS -d 0.0005 -pd $PD -l 2 --repetitions 5

    OMP_PROC_BIND=spread OMP_PLACES=cores python spmv_runahead.py benchmark -v omp-tasks-4 -r $ROWS -c $COLS -d 0.0005 -pd $PD --repetitions 5

 done
