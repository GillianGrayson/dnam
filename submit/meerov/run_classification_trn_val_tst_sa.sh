#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH -o /common/home/yusipov_i/source/DNAmClassMeta/submit/output/%j.txt

code_dir=/common/home/yusipov_i/source/DNAmClassMeta

printf "args:\n $1 \n\n"

srun python $code_dir/run_classification_trn_val_tst_sa.py $1
