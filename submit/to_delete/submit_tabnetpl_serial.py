import os
import numpy as np


check_sum = '121da597d6d3fe7b3b1b22a0ddc26e61'
seed = 2

output_dim = 4
data_path = f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}"

n_d_n_a = [8, 16]
n_steps = [3]
gamma = [1.3, 1.7]
n_independent = [1]
n_shared = [2]
optimizer_lr = [0.01, 0.005, 0.001]
optimizer_weight_decay = [0.0]
scheduler_step_size = [50]
scheduler_gamma = [0.9]

weighted_sampler = True

cpgs_list_origin = "24829"
cpgs_from_model = 'tabnetpl'
cpgs_from_run = 'average'
cpgs_from_variance = 'all'
#counts = np.linspace(10, 500, 50, dtype=int)
counts = [310, 320]

for c in counts:

       project_name = f'{cpgs_from_model}_unnhpc_{cpgs_from_run}_{cpgs_from_variance}_{c}'

       cpgs_fn = f"{data_path}/cpgs/{cpgs_list_origin}/{cpgs_from_model}/{cpgs_from_run}/{cpgs_from_variance}/{c}.xlsx"
       statuses_fn = f"{data_path}/statuses/{output_dim}.xlsx"

       args = f"--multirun project_name={project_name} " \
              f"logger.wandb.offline=True " \
              f"experiment=tabnetpl " \
              f"work_dir=\"{data_path}/models/{cpgs_list_origin}/{project_name}\" " \
              f"data_dir=\"{data_path}\" " \
              f"seed={seed} " \
              f"datamodule.seed={seed} " \
              f"datamodule.path=\"{data_path}\" " \
              f"datamodule.cpgs_fn=\"{cpgs_fn}\" " \
              f"datamodule.statuses_fn=\"{statuses_fn}\" " \
              f"datamodule.weighted_sampler={weighted_sampler} " \
              f"model.input_dim={c} " \
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
