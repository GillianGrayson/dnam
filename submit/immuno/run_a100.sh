#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH -o /common/home/yusipov_i/source/dnam/submit/output/%j.txt

code_dir=/common/home/yusipov_i/source/dnam

printf "args:\n $1 \n\n"

srun python $code_dir/run_regression_trn_val_tst.py $1
