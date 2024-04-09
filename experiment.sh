#!/bin/bash

ROWS=2000
COLS=150000000

python spmv.py benchmark -r $ROWS -c $COLS -d 0.0005 --repetitions 5

for PD in {35..300}; do

    python spmv.py benchmark -o pref-ains -r $ROWS -c $COLS -d 0.0005 --repetitions 5
    python spmv.py benchmark -o pref-spe  -r $ROWS -c $COLS -d 0.0005 --repetitions 5

 done
