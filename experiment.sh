#!/bin/bash

source /home/paul/venv/bin/activate
source opt/intel/oneapi/setvars.sh

# spmv coo on SparseSuite on E-core 8

# HW pref OFF
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream --matrix-format coo profile events
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-ains --matrix-format coo profile events
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-mlir --matrix-format coo profile events

python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-ains --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-mlir --matrix-format coo benchmark --repetitions 10

mv results.json ss-spmv-coo-ecore-prof-and-bench-hw-off.json

# L2 and LLC off
python spmv_on_suitesparse.py -c all --disable-l2-stream --disable-l2-amp --disable-llc-stream --matrix-format coo profile events
python spmv_on_suitesparse.py -c all --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-ains --matrix-format coo profile events
python spmv_on_suitesparse.py -c all --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-mlir --matrix-format coo profile events

python spmv_on_suitesparse.py -c all --disable-l2-stream --disable-l2-amp --disable-llc-stream --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -c all --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-ains --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -c all --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-mlir --matrix-format coo benchmark --repetitions 10

mv results.json ss-spmv-coo-ecore-prof-and-bench-l2-off.json

# Only L1 npp
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l2-stream --disable-l2-amp --disable-llc-stream --matrix-format coo profile events
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-ains --matrix-format coo profile events
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-mlir --matrix-format coo profile events

python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l2-stream --disable-l2-amp --disable-llc-stream --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-ains --matrix-format coo benchmark --repetitions 10
python spmv_on_suitesparse.py -c all --disable-l1-ipp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-mlir --matrix-format coo benchmark --repetitions 10

mv results.json ss-spmv-coo-ecore-prof-and-bench-only-l1-npp.json
