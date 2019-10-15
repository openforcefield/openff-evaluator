#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J emv
#BSUB -W 03:00
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

rm -rf excess_molar_volume && mkdir excess_molar_volume && cd excess_molar_volume
cp ../pure_data_set.json . && cp ../binary_data_set.json .
python ../excess_molar_volume.py &> emv_console_output.log
