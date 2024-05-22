#!/bin/bash

source /home/paul/venv/bin/activate

ROWS=2000
COLS=150000000

python spmv.py benchmark -r $ROWS -c $COLS -d 0.0005 --repetitions 10
python spmv.py benchmark -o pref-ains -r $ROWS -c $COLS -d 0.0005 --repetitions 10

for L2_MSHRS in {1..400}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005 --repetitions 10
done

ROWS=40000
COLS=15000000

python spmv.py benchmark -r $ROWS -c $COLS -d 0.0005 --repetitions 10
python spmv.py benchmark -o pref-ains -r $ROWS -c $COLS -d 0.0005 --repetitions 10

for L2_MSHRS in {1..400}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005 --repetitions 10
done
