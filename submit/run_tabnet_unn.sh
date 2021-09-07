#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH -o /common/home/yusipov_i/source/dnam/submit/output/%j.txt

cudaDevs=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')

for cudaDev in $cudaDevs
do
  echo cudaDev = $cudaDev
done

code_dir=/common/home/yusipov_i/source/dnam

printf "args:\n $1 \n"

srun python $code_dir/run_tabnet.py $1

#srun python $code_dir/run_tabnet.py --multirun hparams_search=tabnet_grid experiment=tabnet work_dir="/home/yusipov_i/data/dnam/datasets/meta/BrainDiseases/variance_0.005/models/tabnet" data_dir="/home/yusipov_i/data/dnam/datasets/meta/BrainDiseases/variance_0.005" datamodule.path="/home/yusipov_i/data/dnam/datasets/meta/BrainDiseases/variance_0.005"
