import os
from glob import glob

data_type = "immuno"
model_type = "elastic_net"
run_type = "trn_val"

model = f"{data_type}_{run_type}_{model_type}"

date_time = "2022-05-19_07-58-43"
data_path = f"/home/yusipov_i/data/unn/immuno/models/{model}/multiruns/{date_time}"

project = f"cluster_{model}"

folders_to_sync = glob(f"{data_path}/*/wandb/offline-run-*")
folders_to_sync.sort()

runs = next(os.walk(data_path))[1]
runs.sort()

for f_id, f in enumerate(folders_to_sync):
    print(f)
    os.system(f"wandb sync --id {runs[f_id]} --project {project} {f}")
