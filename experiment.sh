#!/bin/bash

ROWS=2000
COLS=150000000

source /home/paul/venv/bin/activate

python spmv.py benchmark -r $ROWS -c $COLS -d 0.0005 --repetitions 8

for L1_MSHRS in {1..10}; do
    for L2_MSHRS in {1..100}; do
        python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" -d 0.0005 --repetitions 8
    done
done
