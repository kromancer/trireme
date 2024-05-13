#!/bin/bash

ROWS=40000
COLS=15000000

source /home/paul/venv/bin/activate

for L1_MSHRS in 5 15 30 40; do
    for L2_MSHRS in {1..400}; do
        python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005 --repetitions 10
        python spmv_multistage.py profile events -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005
    done
done
