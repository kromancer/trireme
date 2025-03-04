#!/bin/bash

source /home/paul/venv/bin/activate
source /opt/intel/oneapi/setvars.sh

REPS=10

L2_DD="--l2-stream-dd 192"

MLC_STREAMER="--disable-l1-nlp --disable-l1-ipp --disable-l2-amp --disable-llc-stream"
L1_IPP_MLC_STREAMER="--disable-l1-nlp --disable-l2-amp --disable-llc-stream"
L1_IPP_MLC_STREAMER_LLC_STREAMER="--disable-l1-nlp --disable-l2-amp"

MTX=kmer_U1a

for pd in $(seq 0 4 128); do
    for dd in $(seq 0 5 255); do
        for ovr in {0..15}; do
            python spmv.py --l2-stream-dd $dd --l2-stream-dd-ovr $ovr -1gb -pd=$pd --matrix-format csr benchmark --repetitions $REPS SuiteSparse $MTX
        done
    done
done

mv results.json $MTX-all-enabled-dd-ovr.json

for pd in $(seq 0 4 128); do
    for dd in $(seq 0 5 255); do
        for ovr in {0..15}; do
            python spmv.py --l2-stream-dd $dd --l2-stream-dd-ovr $ovr $MLC_STREAMER -1gb -pd=$pd --matrix-format csr benchmark --repetitions $REPS SuiteSparse $MTX
        done
    done
done

mv results.json $MTX-mlc-streamer-dd-ovr.json

for pd in $(seq 0 4 128); do
    for dd in $(seq 0 5 255); do
        for ovr in {0..15}; do
            python spmv.py --l2-stream-dd $dd --l2-stream-dd-ovr $ovr $L1_IPP_MLC_STREAMER -1gb -pd=$pd --matrix-format csr benchmark --repetitions $REPS SuiteSparse $MTX
        done
    done
done

mv results.json $MTX-l1-ipp-mlc-streamer-dd-ovr.json

for pd in $(seq 0 4 128); do
    for dd in $(seq 0 5 255); do
        for ovr in {0..15}; do
            python spmv.py --l2-stream-dd $dd --l2-stream-dd-ovr $ovr $L1_IPP_MLC_STREAMER_LLC_STREAMER -1gb -pd=$pd --matrix-format csr benchmark --repetitions $REPS SuiteSparse $MTX
        done
    done
done

mv results.json $MTX-l1-ipp-mlc-streamer-llc-streamer-dd-ovr.json
