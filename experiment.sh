#!/bin/bash

ROWS=2000
COLS=150000000
L1_MSHRS=15

source /home/paul/venv/bin/activate


python spmv.py benchmark -r $ROWS -c $COLS -d 0.0005 --repetitions 8
python spmv.py benchmark -r $ROWS -c $COLS --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005 --repetitions 8

for PD in {16..64}; do
    python spmv.py benchmark -o pref-ains -pd "$PD" -r $ROWS -c $COLS -d 0.0005 --repetitions 8
    python spmv.py benchmark -o pref-ains -pd "$PD" -r $ROWS -c $COLS --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005 --repetitions 8
done

for L2_MSHRS in {1..128}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005 --repetitions 8
done

for L2_MSHRS in {1..128}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" -d 0.0005 --repetitions 8
done
