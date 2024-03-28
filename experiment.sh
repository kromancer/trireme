#!/bin/bash

ROWS=2000
COLS=150000000

cd /trireme

source venv/bin/activate

for PD in {35..300}; do

    OMP_PROC_BIND=spread OMP_PLACES=cores python spmv_runahead.py benchmark -v omp-tasks-dummy -r $ROWS -c $COLS -d 0.0005 -pd $PD --repetitions 5

 done
