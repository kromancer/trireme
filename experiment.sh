#!/bin/bash

ROWS=2000
COLS=150000000

source /home/paul/venv/bin/activate

for L2_MSHRS in {20..80}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs 10 --l2-mshrs "$L2_MSHRS" -d 0.0005 --repetitions 8
done
