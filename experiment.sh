#!/bin/bash

ROWS=2000
COLS=150000000

cd /trireme

source venv/bin/activate

for PD in {60..300}; do

    OMP_PROC_BIND=spread OMP_PLACES=cores python spmv_runahead.py benchmark -v omp-tasks-2 -r $ROWS -c $COLS -d 0.0005 -pd $PD -l 2 --repetitions 5

 done
