#!/bin/bash

source /home/paul/venv/bin/activate

python spmv_on_suitesparse.py --matrix-format coo profile events
python spmv_on_suitesparse.py -o pref-ains --matrix-format coo profile events
python spmv_on_suitesparse.py -o pref-mlir --matrix-format coo profile events
