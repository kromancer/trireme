#!/bin/bash

ROWS=2000
COLS=150000000

source /home/paul/venv/bin/activate

for L1_MSHRS in 5 10 15 20 30; do
    for L2_MSHRS in {1..200}; do
        python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005 --repetitions 10
        python spmv_multistage.py profile events -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005
    done
done
