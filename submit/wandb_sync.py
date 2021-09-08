import os
from glob import glob

model = "fcmlp_2"
date_time = "2021-09-07_23-16-07"
folder_path = f"/common/home/yusipov_i/data/dnam/datasets/meta/BrainDiseases/variance_0.005/models/{model}/logs/multiruns/{date_time}"

folders_to_sync = glob(f"{folder_path}/*/wandb/offline-run-*")
folders_to_sync.sort()
for f in folders_to_sync:
    print(f)
    os.system(f"wandb sync {f}")
