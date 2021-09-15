import os
from glob import glob

model = "tabnetpl_unnhpc_1"
project = "tabnetpl_unnhpc_1_learning_rate"
date_time = "2021-09-14_22-36-40"
folder_path = f"/common/home/yusipov_i/data/dnam/datasets/meta/SchizophreniaDepressionParkinson/full/models/{model}/logs/multiruns/{date_time}"

folders_to_sync = glob(f"{folder_path}/*/wandb/offline-run-*")
folders_to_sync.sort()

runs = next(os.walk(folder_path))[1]
runs.sort()

for f_id, f in enumerate(folders_to_sync):
    print(f)
    os.system(f"wandb sync --id {runs[f_id]} --project {project} {f}")