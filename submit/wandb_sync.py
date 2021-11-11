import os
from glob import glob

model = "lightgbm_unnhpc"

check_sum = 'cbfdd0bab9805d1c5d5ecdebec2943dc'
seed = 4
input_dim = 390485
output_dim = 4
date_time = "2021-11-09_15-39-35"
data_path = f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}/models/{model}/logs/multiruns/{date_time}"

project = f"{model}_{seed}_{input_dim}_{output_dim}_{check_sum}_{date_time}"

folders_to_sync = glob(f"{data_path}/*/wandb/offline-run-*")
folders_to_sync.sort()

runs = next(os.walk(data_path))[1]
runs.sort()

for f_id, f in enumerate(folders_to_sync):
    print(f)
    os.system(f"wandb sync --id {runs[f_id]} --project {project} {f}")