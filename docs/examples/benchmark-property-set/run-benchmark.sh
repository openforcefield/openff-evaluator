#!/usr/bin/env bash
#SBATCH -J benchmark
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 5-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --account dmobley_lab_gpu
#SBATCH --output slurm-%x.%A.out

. ~/.bashrc

# Use the right conda environment
conda activate evaluator-test-env

WATER=tip3p
WATER=opc3
WATER=opc

python benchmark.py -n 2000 -w $WATER


