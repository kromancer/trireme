#!/bin/bash

ROWS=2000
COLS=150000000

source /home/paul/venv/bin/activate

python spmv.py benchmark -r $ROWS -c $COLS -d 0.0005 --repetitions 8

for L2_MSHRS in {20..80}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs 10 --l2-mshrs "$L2_MSHRS" -d 0.0005 --disable-l1-ipp --repetitions 8
done

for L2_MSHRS in {20..80}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs 10 --l2-mshrs "$L2_MSHRS" -d 0.0005 --disable-l1-ipp --disable-l1-npp --repetitions 8
done

for L2_MSHRS in {20..80}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs 10 --l2-mshrs "$L2_MSHRS" -d 0.0005 --disable-l1-ipp --disable-l1-npp --disable-l2-stream --repetitions 8
done

for L2_MSHRS in {20..80}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs 10 --l2-mshrs "$L2_MSHRS" -d 0.0005 --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --repetitions 8
done

for L2_MSHRS in {20..80}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs 10 --l2-mshrs "$L2_MSHRS" -d 0.0005 --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream --repetitions 8
done
