import os
from glob import glob

model = "tabnetpl_unnhpc"
input_dim = 380649
output_dim = 8
check_sum = '188aa4fc21423103a3099931bf6c00f0'
date_time = "2021-10-16_01-17-59"
data_path = f"/common/home/yusipov_i/data/dnam/datasets/meta/{check_sum}_{output_dim}/{input_dim}/models/{model}/logs/multiruns/{date_time}"

project = f"{input_dim}_{output_dim}_{check_sum}_{date_time}"

folders_to_sync = glob(f"{data_path}/*/wandb/offline-run-*")
folders_to_sync.sort()

runs = next(os.walk(data_path))[1]
runs.sort()

for f_id, f in enumerate(folders_to_sync):
    print(f)
    os.system(f"wandb sync --id {runs[f_id]} --project {project} {f}")