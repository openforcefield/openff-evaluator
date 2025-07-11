#!/usr/bin/env bash
#SBATCH -J benchmark
#SBATCH -p gpu
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --account dmobley_lab_gpu
#SBATCH --gres=gpu:1

source ~/.bashrc

conda activate evaluator-test-env-openff

python test-preeq.py
