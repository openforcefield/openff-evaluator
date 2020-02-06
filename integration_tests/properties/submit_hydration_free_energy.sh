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
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared:mps=no:

# Enable conda
. ~/.bashrc

# Use the right conda environment
conda activate evaluator
module load cuda/10.1

rm -rf hydration_free_energy && mkdir hydration_free_energy && cd hydration_free_energy
cp ../hydration_data_set.json .
python ../hydration_free_energy.py &> hydration_free_energy.log
