import numpy as np
import pandas as pd
from pathlib import Path


counts = np.linspace(10, 500, 50, dtype=int)

check_sum = '121da597d6d3fe7b3b1b22a0ddc26e61'
path = f"E:/YandexDisk/Work/pydnameth/datasets/meta/{check_sum}"
xai_path = f"{path}/models/xai/logs/runs/2021-11-16_17-08-18"

feat_imp = pd.read_excel(f"{xai_path}/feat_importances.xlsx", index_col="feature")
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
