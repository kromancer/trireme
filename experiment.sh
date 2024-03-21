#!/bin/bash

ROWS=2000
COLS=150000000

cd /trireme

source venv/bin/activate

python spmv.py benchmark -r $ROWS -c $COLS -d 0.0005  --repetitions 10

for PD in {114..500}; do

    OMP_PROC_BIND=spread OMP_PLACES=cores python spmv_runahead.py benchmark -r $ROWS -c $COLS -d 0.0005 -pd $PD --repetitions 5
    OMP_PROC_BIND=spread OMP_PLACES=cores python spmv_runahead.py benchmark -v omp-tasks-2 -r $ROWS -c $COLS -d 0.0005 -pd $PD --repetitions 5

 done
