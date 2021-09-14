import os

project_name = 'tabnetpl_unnhpc_1'

data_path = "/home/yusipov_i/data/dnam/datasets/meta/SchizophreniaDepressionParkinson/full"

n_d_n_a = [8, 16]
n_steps = [3, 6]
gamma = [1.3, 1.7]
n_independent = [1, 2]
n_shared = [2, 4]
optimizer_lr = [0.0001, 0.0005, 0.001]
optimizer_weight_decay = [0.0]
scheduler_step_size = [150]
scheduler_gamma = [0.9]

args = f"--multirun project_name={project_name} " \
       f"logger.wandb.offline=True " \
       f"experiment=tabnetpl " \
       f"work_dir=\"{data_path}/models/{project_name}\" " \
       f"data_dir=\"{data_path}\" " \
       f"datamodule.path=\"{data_path}\" " \
       f"model.mask_type=\"sparsemax\" " \
       f"model.n_d_n_a={','.join(str(x) for x in n_d_n_a)} " \
       f"model.n_steps={','.join(str(x) for x in n_steps)} " \
       f"model.gamma={','.join(str(x) for x in gamma)} " \
       f"model.n_independent={','.join(str(x) for x in n_independent)} " \
       f"model.n_shared={','.join(str(x) for x in n_shared)} " \
       f"model.optimizer_lr={','.join(str(x) for x in optimizer_lr)} " \
       f"model.optimizer_weight_decay={','.join(str(x) for x in optimizer_weight_decay)} " \
       f"model.scheduler_step_size={','.join(str(x) for x in scheduler_step_size)} " \
       f"model.scheduler_gamma={','.join(str(x) for x in scheduler_gamma)} "

os.system(f"sbatch run_tabnetpl_unn.sh \"{args}\"")
