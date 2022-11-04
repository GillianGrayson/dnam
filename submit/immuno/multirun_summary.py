import pandas as pd
from glob import glob
import os
from omegaconf import OmegaConf

path_models = f"/common/home/yusipov_i/data/unn/immuno/models"

model = 'elastic_net'

path_load = f"{path_models}/immuno_trn_val_{model}"

files = glob(f"{path_load}/multiruns/*/*/metrics_val_best_*.xlsx")

df_tmp = pd.read_excel(files[0], index_col="metric")
head, tail = os.path.split(files[0])
cfg = OmegaConf.load(f"{head}/.hydra/overrides.yaml")
params = []
for param_pair in cfg:
    param, val = param_pair.split('=')
    params.append(param)
df_res = pd.DataFrame(index=files, columns=[m + "_trn" for m in df_tmp.index.values] + [m + "_val" for m in df_tmp.index.values] + params)
for file in files:
    # Validation
    df_val = pd.read_excel(file, index_col="metric")
    for metric in df_val.index.values:
        df_res.at[file, metric + "_val"] = df_val.at[metric, "val"]

    # Train
    head, tail = os.path.split(file)
    tail = tail.replace('val', 'trn')
    df_trn = pd.read_excel(f"{head}/{tail}", index_col="metric")
    for metric in df_trn.index.values:
        df_res.at[file, metric + "_trn"] = df_trn.at[metric, "trn"]

    # Params
    cfg = OmegaConf.load(f"{head}/.hydra/overrides.yaml")
    for param_pair in cfg:
        param, val = param_pair.split('=')
        df_res.at[file, param] = val

first_columns = [
    'mean_absolute_error_trn',
    'mean_absolute_error_cv_mean_trn',
    'mean_absolute_error_val',
    'mean_absolute_error_cv_mean_val',
    'mean_absolute_error_tst',
    'mean_absolute_error_cv_mean_tst',
    'mean_absolute_error_cv_mean_val_tst_val'
]
df_res = df_res[first_columns + [col for col in df_res.columns if col not in first_columns]]
df_res.to_excel(f"{path_load}/summary.xlsx", index=True, index_label="file")