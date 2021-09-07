import os

project_name = 'tabnet_2'

data_path = "/home/yusipov_i/data/dnam/datasets/meta/BrainDiseases/variance_0.005"

n_d = [8, 16, 32]
n_a = [8, 16, 32]
n_steps = [3, 6, 9]
gamma = [1.3, 1.5, 1.7]
n_independent = [1, 2, 4]
n_shared = [1, 2, 4]
momentum = [0.01, 0.02, 0.05]
lambda_sparse  = [0.0001, 0.001, 0.01]
optimizer_lr = [0.00001, 0.0001, 0.001, 0.01]
optimizer_weight_decay = [0.0, 0.0001, 0.001]

args = f"--multirun project_name={project_name} " \
       f"hparams_search=tabnet logger.wandb.offline=True " \
       f"experiment=tabnet work_dir=\"{data_path}/models/{project_name}\" " \
       f"data_dir=\"{data_path}\" " \
       f"datamodule.path=\"{data_path}\""

os.system(f"sbatch run_tabnet_unn.sh \"{args}\"")
