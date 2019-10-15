#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J density
#BSUB -W 01:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any gpu options.
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared:mps=no:

# Enable conda
. ~/.bashrc

# Use the right conda environment
conda activate propertyestimator
module load cuda/10.1

rm -rf density && mkdir density && cd density
cp ../pure_data_set.json . && cp ../binary_data_set.json .
python ../density.py &> density_console_output.log