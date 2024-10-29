#!/bin/bash

source /home/paul/venv/bin/activate

# spmv coo on SparseSuite on E-core 8

# Profile with HW prefetching off
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream --matrix-format coo profile events
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-ains --matrix-format coo profile events
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-mlir --matrix-format coo profile events

# Benchmark with HW prefetching off
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-ains --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-mlir --matrix-format coo benchmark --repetitions 10
