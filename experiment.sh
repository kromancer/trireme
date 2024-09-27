#!/bin/bash

source /home/paul/venv/bin/activate

# spmv coo on SparseSuite on E-core 8

# Profile with HW prefetching
python spmv_on_suitesparse.py -o pref-ains --matrix-format coo profile events
python spmv_on_suitesparse.py -o pref-mlir --matrix-format coo profile events
python spmv_on_suitesparse.py --matrix-format coo profile events

# Benchmark with HW prefetching
python spmv_on_suitesparse.py --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -o pref-ains --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -o pref-mlir --matrix-format coo benchmark --repetitions 10

