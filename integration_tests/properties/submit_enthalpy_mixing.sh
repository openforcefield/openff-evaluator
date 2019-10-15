#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J eom
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

rm -rf enthalpy_of_mixing && mkdir enthalpy_of_mixing && cd enthalpy_of_mixing
cp ../pure_data_set.json . && cp ../binary_data_set.json .
python ../enthalpy_of_mixing.py &> eom_console_output.log
