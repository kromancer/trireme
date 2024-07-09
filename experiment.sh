#!/bin/bash

source /home/paul/venv/bin/activate

# For density 2^-13, the matrix will have ~2.5GB of values, ~36 per row
# For density 2^-18, the matrix will have ~79MB of values, ~1 per row
ROWS=300000

# The vector will be around 60MB ~2 x L3 cache
COLS=9000000

# START and END exponents
START=-13
END=-18

i=$START
while (( $(echo "$i >= $END" | bc -l) )); do
    DENS=$(awk "BEGIN {print 2^$i}")
    python spmv.py benchmark -i $ROWS -j $COLS -d $DENS --repetitions 10
    python spmv.py benchmark -o pref-ains -i $ROWS -j $COLS -d $DENS --repetitions 10 --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream
    python spmv.py benchmark -o pref-mlir -i $ROWS -j $COLS -d $DENS --repetitions 10 --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream
    i=$(echo "$i - 0.5" | bc)
done
