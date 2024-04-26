#!/bin/bash

ROWS=2000
COLS=150000000
L1_MSHRS=15

source /home/paul/venv/bin/activate

for L2_MSHRS in {1..128}; do
    python spmv_multistage.py profile memory-access -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005
done
