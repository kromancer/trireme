#!/bin/bash

source /home/paul/venv/bin/activate

# spmv coo on SparseSuite on E-core 8

# Benchmark with HW prefetching
# python spmv_on_suitesparse.py --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -c pref_ains_benchmark -o pref-ains --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -c pref_mlir_benchmark -o pref-mlir --matrix-format coo benchmark --repetitions 10
