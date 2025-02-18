#!/bin/bash

swapoff -a
fallocate -l 64G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
sysctl -w vm.overcommit_memory=1
sysctl vm.swappiness=80

source /home/paul/venv/bin/activate
apptainer exec --memory 120G --memory-swap 184G [path to apptainer image]/trireme.sif sh -c "python convert_ss_to_csr.py"
