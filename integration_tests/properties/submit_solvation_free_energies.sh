#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J g_solv
#BSUB -W 72:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any gpu options.
#BSUB -q cpuqueue

# Enable conda
. ~/.bashrc

# Use the right conda environment
conda activate propertyestimator
module load cuda/10.1

rm -rf solvation_free_energies && mkdir solvation_free_energies && cd solvation_free_energies
cp ../pure_data_set.json . && cp ../binary_data_set.json .
python ../solvation_free_energies.py &> g_solv_console_output.log
