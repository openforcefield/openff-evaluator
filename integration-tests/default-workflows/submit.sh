#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J defaults
#BSUB -W 48:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
#BSUB -q cpuqueue

# Enable conda
. /home/boothros/.bashrc
conda activate openff-evaluator-XXX

conda env export > environment.yml

# Launch my program.
module load cuda/10.0
python run.py
