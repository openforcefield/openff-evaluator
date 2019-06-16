#!/usr/bin/env bash
export CUDA_HOME=/usr/local/cuda-9.1/
export CUDA_INCLUDE_DIR=/usr/local/cuda-9.1/include
export CUDA_LIB_DIR=/usr/local/cuda-9.1/lib64
export OPENMM_CUDA_COMPILER=/usr/local/cuda-9.1/bin/nvcc
export PATH=/usr/local/cuda-9.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:$LD_LIBRARY_PATH
export OE_LICENSE="$HOME/oe_license.txt"

conda activate openmm
