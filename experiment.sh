#!/bin/bash

source /home/paul/venv/bin/activate
source /opt/intel/oneapi/setvars.sh

ROWS=200000
PROJ_DIR="/home/paul/advi_results"

SNAPSHOT="/home/paul/baseline"
N_PROJ_DIR="/home/paul/advi_baseline"
python spmv.py --matrix-format csr profile advisor roofline synthetic -i $ROWS -j 100000000 --dtype float64 --density 0.00005
advisor --snapshot --pack --cache-sources --cache-binaries --project-dir=$PROJ_DIR -- $SNAPSHOT
mv $PROJ_DIR $N_PROJ_DIR

SNAPSHOT="/home/paul/baseline_off"
N_PROJ_DIR="/home/paul/advi_baseline_off"
python spmv.py --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream --matrix-format csr profile advisor roofline synthetic -i $ROWS -j 100000000 --dtype float64 --density 0.00005
advisor --snapshot --pack --cache-sources --cache-binaries --project-dir=$PROJ_DIR -- $SNAPSHOT
mv $PROJ_DIR $N_PROJ_DIR

SNAPSHOT="/home/paul/mlir_pref"
N_PROJ_DIR="/home/paul/advi_mlir_pref"
python spmv.py -o pref-simple --matrix-format csr profile advisor roofline synthetic -i $ROWS -j 100000000 --dtype float64 --density 0.00005
advisor --snapshot --pack --cache-sources --cache-binaries --project-dir=$PROJ_DIR -- $SNAPSHOT
mv $PROJ_DIR $N_PROJ_DIR

SNAPSHOT="/home/paul/mlir_pref_off"
N_PROJ_DIR="/home/paul/advi_mlir_pref_off"
python spmv.py --disable-l1-ipp --disable-l1-npp --disable-l2-stream --disable-l2-amp --disable-llc-stream -o pref-simple --matrix-format csr profile advisor roofline synthetic -i $ROWS -j 100000000 --dtype float64 --density 0.00005
advisor --snapshot --pack --cache-sources --cache-binaries --project-dir=$PROJ_DIR -- $SNAPSHOT
mv $PROJ_DIR $N_PROJ_DIR

SNAPSHOT="/home/paul/ains"
N_PROJ_DIR="/home/paul/advi_ains"
python spmv.py -o pref-ains --matrix-format csr profile advisor roofline synthetic -i $ROWS -j 100000000 --dtype float64 --density 0.00005
advisor --snapshot --pack --cache-sources --cache-binaries --project-dir=$PROJ_DIR -- $SNAPSHOT
mv $PROJ_DIR $N_PROJ_DIR
