import os
from glob import glob

model = "tabnetpl_unnhpc"

check_sum = 'd11b5f9b6efd089db42a3d5e6b375430'
input_dim = 375614
output_dim = 6
date_time = "2021-10-16_01-17-59"
data_path = f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}/models/{model}/logs/multiruns/{date_time}"

project = f"{input_dim}_{output_dim}_{check_sum}_{date_time}"

folders_to_sync = glob(f"{data_path}/*/wandb/offline-run-*")
folders_to_sync.sort()

runs = next(os.walk(data_path))[1]
runs.sort()

for f_id, f in enumerate(folders_to_sync):
    print(f)
    os.system(f"wandb sync --id {runs[f_id]} --project {project} {f}")