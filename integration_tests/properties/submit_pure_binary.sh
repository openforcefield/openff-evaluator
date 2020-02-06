#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J all_prop
#BSUB -W 05:59
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
conda activate evaluator
module load cuda/10.1

rm -rf all && mkdir all && cd all
cp ../pure_data_set.json . && cp ../binary_data_set.json .
python ../all_properties.py &> all_properties_console_output.log