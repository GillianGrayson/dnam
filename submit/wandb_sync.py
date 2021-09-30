import os
from glob import glob

num_classes = 3
model = "tabnetpl_unnhpc"
date_time = "2021-09-30_16-11-30"
data_type = '17'
project = f"{num_classes}_{data_type}_{model}_{date_time}"
folder_path = f"/common/home/yusipov_i/data/dnam/datasets/meta/SchizophreniaDepressionParkinsonCases/{data_type}/models/{model}/logs/multiruns/{date_time}"

folders_to_sync = glob(f"{folder_path}/*/wandb/offline-run-*")
folders_to_sync.sort()

runs = next(os.walk(folder_path))[1]
runs.sort()

for f_id, f in enumerate(folders_to_sync):
    print(f)
    os.system(f"wandb sync --id {runs[f_id]} --project {project} {f}")