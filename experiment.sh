#!/bin/bash

source /home/paul/venv/bin/activate

ROWS=200000
COLS=100000000

python spmv.py -o no-opt --matrix-format csr benchmark --repetitions 10 synthetic -i $ROWS -j $COLS --dtype float64 --density 0.00005

for i in {15..64}
do
    python spmv.py -o pref-ains  -pd $i --matrix-format csr benchmark --repetitions 10 synthetic -i $ROWS -j $COLS --dtype float64 --density 0.00005
    python spmv.py -o pref-mlir  -pd $i --matrix-format csr benchmark --repetitions 10 synthetic -i $ROWS -j $COLS --dtype float64 --density 0.00005
    python spmv.py -o pref-split -pd $i --matrix-format csr benchmark --repetitions 10 synthetic -i $ROWS -j $COLS --dtype float64 --density 0.00005
done
