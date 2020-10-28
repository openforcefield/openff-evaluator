#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J paprika
#BSUB -W 24:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any gpu options.
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared:mps=no:

# Enable conda
. /home/boothros/.bashrc
conda activate openff-evaluator-XXX

conda env export > environment.yml

# Launch my program.
module load cuda/10.0
python run.py
