#!/bin/bash

source /home/paul/venv/bin/activate
source /opt/intel/oneapi/setvars.sh

export OMP_NUM_THREADS=5
export LD_LIBRARY_PATH=/opt/llvm/lib:/opt/llvm/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH

ROWS=200000
PROJ_DIR="/home/paul/advi_results"

SNAPSHOT="/home/paul/baseline_omp_5"
N_PROJ_DIR="/home/paul/advi_baseline_omp_5"
python spmv.py -o omp --matrix-format csr profile advisor roofline synthetic -i $ROWS -j 100000000 --dtype float64 --density 0.00005
advisor --snapshot --pack --cache-sources --cache-binaries --project-dir=$PROJ_DIR -- $SNAPSHOT
mv $PROJ_DIR $N_PROJ_DIR


SNAPSHOT="/home/paul/mlir_pref_omp_5"
N_PROJ_DIR="/home/paul/advi_mlir_pref_omp_5"
python spmv.py -o pref-mlir-omp --matrix-format csr profile advisor roofline synthetic -i $ROWS -j 100000000 --dtype float64 --density 0.00005
advisor --snapshot --pack --cache-sources --cache-binaries --project-dir=$PROJ_DIR -- $SNAPSHOT
mv $PROJ_DIR $N_PROJ_DIR
