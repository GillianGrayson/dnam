#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH -o /common/home/yusipov_i/source/dnam/submit/output/%j.txt

code_dir=/common/home/yusipov_i/source/dnam

srun python $code_dir/run_tabnet.py --multirun hparams_search=tabnet_grid experiment=tabnet work_dir="/common/home/yusipov_i/data/dnam/datasets/meta/BrainDiseases/variance(0.005)/models/tabnet" data_dir="/common/home/yusipov_i/data/dnam/datasets/meta/BrainDiseases/variance(0.005)" datamodule.path="/common/home/yusipov_i/data/dnam/datasets/meta/BrainDiseases/variance(0.005)"
