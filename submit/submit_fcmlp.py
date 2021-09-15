import os

project_name = 'fcmlp_unnhpc_1'

data_path = "/home/yusipov_i/data/dnam/datasets/meta/SchizophreniaDepressionParkinson/full"
n_input = 391023

topology  = [[512, 256, 128], [512, 256], [256, 128]]
lr = [0.00001, 0.0001, 0.001, 0.01]
weight_decay = [0.0, 0.0001, 0.001]

weighted_sampler = True

args = f"--multirun project_name={project_name} " \
       f"hparams_search=fcmlp logger.wandb.offline=True " \
       f"experiment=fcmlp work_dir=\"{data_path}/models/{project_name}\" " \
       f"data_dir=\"{data_path}\" " \
       f"datamodule.path=\"{data_path}\"" \
       f"datamodule.weighted_sampler={weighted_sampler} " \
       f"model.n_input={n_input} "

os.system(f"sbatch run_fcmlp_unn.sh \"{args}\"")
