#!/usr/bin/env bash
#SBATCH -J equilibrate
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

python equilibrate.py -n 2000
