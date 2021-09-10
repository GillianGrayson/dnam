import os
from glob import glob

model = "tabnet_3"
date_time = "2021-09-09_00-41-41"
folder_path = f"/common/home/yusipov_i/data/dnam/datasets/meta/BrainDiseases/variance_0.005/models/{model}/logs/multiruns/{date_time}"

folders_to_sync = glob(f"{folder_path}/*/wandb/offline-run-*")
folders_to_sync.sort()

runs = next(os.walk(folder_path))[1]
runs.sort()

for f_id, f in enumerate(folders_to_sync):
    print(f)
    os.system(f"wandb sync --id {runs[f_id]} --project {model} {f}")