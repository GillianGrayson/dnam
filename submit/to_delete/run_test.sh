#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH -o /common/home/yusipov_i/source/dnam/submit/output/%j.txt

code_dir=/common/home/yusipov_i/source/dnam

srun python $code_dir/run_test.py