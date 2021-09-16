import os
from glob import glob

model = "fcmlp_unnhpc_1"
date_time = "2021-09-16_01-29-49"
data_type = 'full'
project = f"{data_type}_{model}_{date_time}"
folder_path = f"/common/home/yusipov_i/data/dnam/datasets/meta/SchizophreniaDepressionParkinson/{data_type}/models/{model}/logs/multiruns/{date_time}"

folders_to_sync = glob(f"{folder_path}/*/wandb/offline-run-*")
folders_to_sync.sort()

runs = next(os.walk(folder_path))[1]
runs.sort()

for f_id, f in enumerate(folders_to_sync):
    print(f)
    os.system(f"wandb sync --id {runs[f_id]} --project {project} {f}")