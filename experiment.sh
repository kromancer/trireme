#!/bin/bash

source /home/paul/venv/bin/activate
source /opt/intel/oneapi/setvars.sh

ROWS=200000
COLS=100000000

python spmv.py -o no-opt --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream --matrix-format csr profile vtune events synthetic -i $ROWS -j $COLS --dtype float64 --density 0.00005
