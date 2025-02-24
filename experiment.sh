#!/bin/bash

source /home/paul/venv/bin/activate
source /opt/intel/oneapi/setvars.sh

DISABLE_ALL="--disable-l1-nlp --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream"
DISABLE_OPT="--disable-l1-nlp --disable-l2-amp"

python spmv_on_suitesparse.py -c all -1gb $DISABLE_OPT -o pref-ains --matrix-format csr profile perf record
mv results.json prof-ains.json

python spmv_on_suitesparse.py -c all -1gb $DISABLE_OPT -o pref-mlir --matrix-format csr profile perf record
mv results.json prof-mlir.json

python spmv_on_suitesparse.py -c all -1gb -o no-opt --matrix-format csr profile perf record
mv results.json prof-no-opt.json

python spmv_on_suitesparse.py -c all -1gb $DISABLE_OPT -o pref-ains --matrix-format csr benchmark --repetitions 10
mv results.json bench-ains.json

python spmv_on_suitesparse.py -c all -1gb $DISABLE_OPT -o pref-mlir --matrix-format csr benchmark --repetitions 10
mv results.json bench-mlir.json

python spmv_on_suitesparse.py -c all -1gb -o no-opt --matrix-format csr benchmark --repetitions 10
mv results.json bench-no-opt.json
