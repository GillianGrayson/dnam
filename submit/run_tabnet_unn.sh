#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH -o /common/home/yusipov_i/source/dnam/submit/output/%j.txt

module load cuda/cuda-11.3
export CUDA_VISIBLE_DEVICES=0

code_dir=/common/home/yusipov_i/source/dnam

printf "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES \n"
printf "args:\n $1 \n"

srun python $code_dir/run_tabnet.py $1