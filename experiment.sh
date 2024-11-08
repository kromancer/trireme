#!/bin/bash

source /home/paul/venv/bin/activate

# spmv coo on SparseSuite on E-core 8

# Profile with HW prefetching on
python spmv_on_suitesparse.py -c no-opt-prof --matrix-format coo profile events

# Benchmark with HW prefetching on
python spmv_on_suitesparse.py -c no-opt-bench --matrix-format coo benchmark --repetitions 10
