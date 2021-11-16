import os

project_name = 'lightgbm_unnhpc'

check_sum = '121da597d6d3fe7b3b1b22a0ddc26e61'
input_dim = 390485
output_dim = 4
data_path = f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}"

cpgs_fn = f"{data_path}/cpgs/{input_dim}.xlsx"
statuses_fn = f"{data_path}/statuses/{output_dim}.xlsx"
weighted_sampler = True

seed = 1

learning_rate = [0.05, 0.005]
num_leaves = [31, 63]
min_data_in_leaf = [10, 20, 40]
feature_fraction = [0.9]
bagging_fraction = [0.8]

args = f"--multirun project_name={project_name} " \
       f"logger.wandb.offline=True " \
       f"experiment=lightgbm " \
       f"work_dir=\"{data_path}/models/{project_name}\" " \
       f"data_dir=\"{data_path}\" " \
       f"seed={seed} " \
       f"datamodule.seed={seed} " \
       f"datamodule.path=\"{data_path}\" " \
       f"datamodule.cpgs_fn=\"{cpgs_fn}\" " \
       f"datamodule.statuses_fn=\"{statuses_fn}\" " \
       f"datamodule.weighted_sampler={weighted_sampler} " \
       f"model.input_dim={input_dim} " \
       f"model.output_dim={output_dim} " \
       f"model.learning_rate={','.join(str(x) for x in learning_rate)} " \
       f"model.num_leaves={','.join(str(x) for x in num_leaves)} " \
       f"model.min_data_in_leaf={','.join(str(x) for x in min_data_in_leaf)} " \
       f"model.feature_fraction={','.join(str(x) for x in feature_fraction)} " \
       f"model.bagging_fraction={','.join(str(x) for x in bagging_fraction)} "

os.system(f"sbatch run_lightgbm_unn.sh \"{args}\"")
