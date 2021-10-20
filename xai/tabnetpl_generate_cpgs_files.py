import numpy as np
import pandas as pd
from pathlib import Path


counts = np.linspace(10, 200, 20, dtype=int)

check_sum = 'd11b5f9b6efd089db42a3d5e6b375430'
path = f"E:/YandexDisk/Work/pydnameth/datasets/meta/{check_sum}"
xai_path = f"{path}/models/xai/logs/runs/2021-10-20_19-39-23"

feat_imp = pd.read_excel(f"{xai_path}/feat_importances.xlsx", index_col="feat")
feat_imp.index.name = "CpG"
feat_imp.sort_values(['average'], ascending=[False], inplace=True)

Path(f"{path}/cpgs/all").mkdir(parents=True, exist_ok=True)
for c in counts:
    curr_feat_imp = feat_imp.loc[feat_imp.index.values[0:c], ['average', 'variance']]
    curr_feat_imp.to_excel(f"{path}/cpgs/all/{c}.xlsx", index=True)

thresholds = [0.001, 0.005, 0.01]
for th in thresholds:
    feat_imp_th = feat_imp.loc[feat_imp['variance'] > th, :]
    feat_imp_th.sort_values(['average'], ascending=[False], inplace=True)
    Path(f"{path}/cpgs/{th}").mkdir(parents=True, exist_ok=True)
    for c in counts:
        curr_feat_imp = feat_imp.loc[feat_imp_th.index.values[0:c], ['average', 'variance']]
        curr_feat_imp.to_excel(f"{path}/cpgs/{th}/{c}.xlsx", index=True)
