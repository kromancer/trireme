#!/bin/bash

for rows in {1..200}; do
    taskset -c 0,1,2,3,4,5,6,7 python spmv.py benchmark -o pref-ains -r $rows -c 2000000000 -d 0.0005 --repetitions 10 -pd 64 -l 2
    taskset -c 0,1,2,3,4,5,6,7 python spmv.py profile toplev -o pref-ains -r $rows -c 2000000000 -d 0.0005 -pd 64 -l 2
    taskset -c 0,1,2,3,4,5,6,7 python spmv.py profile events -o pref-ains -r $rows -c 2000000000 -d 0.0005 -pd 64 -l 2
done
