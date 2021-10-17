import os

project_name = 'tabnetpl_unnhpc'

check_sum = '56727e67034ec72e6e3e66a00f1f841a'
input_dim = 380649
output_dim = 7
data_path = f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}_{output_dim}/{input_dim}"

n_d_n_a = [8, 16]
n_steps = [3]
gamma = [1.3, 1.7]
n_independent = [1]
n_shared = [2]
optimizer_lr = [0.05, 0.01, 0.005, 0.001, 0.0005]
optimizer_weight_decay = [0.0]
scheduler_step_size = [50]
scheduler_gamma = [0.9]

weighted_sampler = True

args = f"--multirun project_name={project_name} " \
       f"logger.wandb.offline=True " \
       f"experiment=tabnetpl " \
       f"work_dir=\"{data_path}/models/{project_name}\" " \
       f"data_dir=\"{data_path}\" " \
       f"datamodule.path=\"{data_path}\" " \
       f"datamodule.weighted_sampler={weighted_sampler} " \
       f"model.input_dim={input_dim} " \
       f"model.output_dim={output_dim} " \
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
