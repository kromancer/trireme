#!/bin/bash

source /home/paul/venv/bin/activate
source /opt/intel/oneapi/setvars.sh

export PKG_CONFIG_PATH=/opt/intel/oneapi/vtune/latest/sdk/include/pkgconfig/lib64:$PKG_CONFIG_PATH
export LC_ALL=C
export OMP_PLACES=cores
export OMP_PROC_BIND=true

dd="16"
ovr="9"
xq_thres="4"

L1_IPP_MLC_STREAMER_MLC_AMP_LLC_STREAMER="--disable-l1-nlp"
L1_IPP_MLC_STREAMER_LLC_STREAMER="--disable-l1-nlp --disable-l2-amp"
DISABLE_ALL="--disable-l1-nlp --disable-l1-npp --disable-l1-ipp --disable-l2-stream --disable-l2-amp --disable-llc-stream"
k=2
reps=10
pd=45
MTX=wikipedia-20070206



########################
# spmv
########################

# Small

python kernel_on_suitesparse.py -c all --kernel spmv -o no-opt            -1gb $L1_IPP_MLC_STREAMER_LLC_STREAMER --matrix-format csr --rep-file ./no-opt.json    benchmark --repetitions $reps
rm -rf workdir-*

python kernel_on_suitesparse.py -c all --kernel spmm -o pref-mlir -pd $pd -1gb $L1_IPP_MLC_STREAMER_LLC_STREAMER --matrix-format csr --rep-file ./pref-mlir.json benchmark --repetitions $reps
rm -rf workdir-*

python kernel_on_suitesparse.py -c all --kernel spmv -o pref-ains -pd $pd -1gb $L1_IPP_MLC_STREAMER_LLC_STREAMER --matrix-format csr --rep-file ./pref-ains.json benchmark --repetitions $reps
rm -rf workdir-*


# Big

python kernel_on_suitesparse.py -c big --kernel spmv -o no-opt            -1gb $L1_IPP_MLC_STREAMER_LLC_STREAMER --matrix-format csr --rep-file ./no-opt.json    benchmark --repetitions $reps
rm -rf workdir-*

python kernel_on_suitesparse.py -c big --kernel spmv -o pref-mlir -pd $pd -1gb $L1_IPP_MLC_STREAMER_LLC_STREAMER --matrix-format csr --rep-file ./pref-mlir.json benchmark --repetitions $reps
rm -rf workdir-*

python kernel_on_suitesparse.py -c big --kernel spmv -o pref-ains -pd $pd -1gb $L1_IPP_MLC_STREAMER_LLC_STREAMER --matrix-format csr --rep-file ./pref-ains.json benchmark --repetitions $reps
rm -rf workdir-*

########################
# spmm
########################

# Small

python kernel_on_suitesparse.py -c all --kernel spmm -o no-opt            -1gb $L1_IPP_MLC_STREAMER_MLC_AMP_LLC_STREAMER --matrix-format csr --rep-file ./spmm-no-opt.json  benchmark --repetitions $reps
rm -rf workdir-*

python kernel_on_suitesparse.py -c all --kernel spmm -o pref-mlir -pd $pd -1gb $L1_IPP_MLC_STREAMER_MLC_AMP_LLC_STREAMER --matrix-format csr --rep-file ./spmm-pref-mlir.json benchmark --repetitions $reps
rm -rf workdir-*


# Big

python kernel_on_suitesparse.py -c big --kernel spmm -o no-opt            -1gb $L1_IPP_MLC_STREAMER_LLC_STREAMER --matrix-format csr --rep-file ./spmm-no-opt.json benchmark --repetitions $reps
rm -rf workdir-*

python kernel_on_suitesparse.py -c big --kernel spmm -o pref-mlir -pd $pd -1gb $L1_IPP_MLC_STREAMER_LLC_STREAMER --matrix-format csr --rep-file ./spmm-pref-mlir.json benchmark --repetitions $reps
rm -rf workdir-*
