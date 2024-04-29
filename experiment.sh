#!/bin/bash

ROWS=2000
COLS=150000000

source /home/paul/venv/bin/activate

L1_MSHRS=15
L2_MSHRS=98

python spmv_multistage.py profile uarch-exploration -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005
python spmv_multistage.py profile memory-access -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005

for L2_MSHRS in {99..150}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005 --repetitions 10
    python spmv_multistage.py profile uarch-exploration -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005
    python spmv_multistage.py profile memory-access -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005
done

L1_MSHRS=20

for L2_MSHRS in {1..150}; do
    python spmv_multistage.py benchmark -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005 --repetitions 10
    python spmv_multistage.py profile uarch-exploration -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005
    python spmv_multistage.py profile memory-access -r $ROWS -c $COLS --l1-mshrs "$L1_MSHRS" --l2-mshrs "$L2_MSHRS" --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -d 0.0005
done
