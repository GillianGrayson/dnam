import os
from pathlib import Path
import numpy as np
import pandas as pd

disease = "Parkinson"
data_type = "non_harmonized"
model_type = "lightgbm"
run_type = "trn_val_tst"

tst_dataset = "GSE72774"

in_dim = 43019

hparams_type = "seed"

optimized_part = "val"

base_dir = f"/common/home/yusipov_i/data/dnam/datasets/meta/GPL13534_Blood/{disease}"

project_name = f'{disease}_{data_type}_{run_type}_{tst_dataset}_{model_type}_{hparams_type}_{optimized_part}'

args = f"--multirun " \
       f"disease={disease} " \
       f"data_type={data_type} " \
       f"model_type={model_type} " \
       f"project_name={project_name} " \
       f"tst_dataset={tst_dataset} " \
       f"in_dim={in_dim} " \
       f"logger=none " \
       f"logger.wandb.offline=True " \
       f"base_dir={base_dir} " \
       f"experiment=dnam/classification/{run_type}/sa " \
       f"hparams_search=dnam/classification/{model_type}/{hparams_type} " \

os.system(f"sbatch run_multiclass_trn_val_tst_sa.sh \"{args}\"")
