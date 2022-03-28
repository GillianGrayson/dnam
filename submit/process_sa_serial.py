import os
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.layout import add_layout
import plotly.express as px


model_sa = 'catboost'

num_realizations = 8
n_feats = np.linspace(10, 500, 50, dtype=int)

base_dir = "/common/home/yusipov_i/data/dnam/datasets/meta/GPL13534_Blood/Schizophrenia"
models_dir = f"{base_dir}/harmonized/models"

metrics = [
    "accuracy_weighted",
    "f1_weighted",
    "cohen_kappa",
    "matthews_corrcoef",
    "auroc_weighted",
]
optimized_metric = "accuracy_weighted"
direction = "max"

baseline_fn = f"{base_dir}/harmonized/models/baseline/dnam_harmonized_multiclass_Status_trn_val_tst_{model_sa}/runs/2022-03-27_01-51-36/metrics_val_best_0013.xlsx"
baseline_metrics_df = pd.read_excel(baseline_fn, index_col="metric")

metrics_global_df = pd.DataFrame(index=n_feats, columns=metrics)
metrics_global_df.index.name = "n_feat"
for n_feat in n_feats:
    print(n_feat)
    project_name = f'dnam_harmonized_multiclass_Status_trn_val_tst_{model_sa}_{n_feat}'
    files = glob(f"{models_dir}/{project_name}/multiruns/*/*/metrics_val_best_*.xlsx")

    if len(files) != num_realizations:
        print(len(files))
        print(f"Available files for {n_feat}:")
        for f in files:
            print(f)
        raise ValueError("Some files are missed!")

    metrics_local_df = pd.DataFrame(index=files, columns=[optimized_metric])
    for file in files:
        df = pd.read_excel(file, index_col="metric")
        metrics_local_df.at[file, optimized_metric] = df.at[optimized_metric, "val"]
    metrics_local_df.sort_values([optimized_metric], ascending=[False if direction == "max" else True], inplace=True)
    best_file = metrics_local_df.index.values[0]
    print(best_file)
    df = pd.read_excel(best_file, index_col="metric")
    for m in metrics:
        metrics_global_df.at[n_feat, m] = df.at[m, "val"]

Path(f"{models_dir}/{model_sa}_iterative").mkdir(parents=True, exist_ok=True)
metrics_global_df.to_excel(f"{models_dir}/{model_sa}_iterative/metrics.xlsx", index=True)
for m in metrics:
    fig = go.Figure()
    add_scatter_trace(fig, metrics_global_df.index.values, metrics_global_df.loc[:, m], f"", mode='lines+markers')
    fig.add_trace(
        go.Scatter(
            x=[n_feats[0], n_feats[-1]],
            y=[-baseline_metrics_df.at[m, "val"], baseline_metrics_df.at[m, "val"]],
            showlegend=False,
            mode='lines',
            line=dict(color='black', width=3, dash='dash')
        )
    )
    fig.update_layout({'colorway': ['red', 'black']})
    add_layout(fig, f"Number of features in model", f"{m}", "")
    save_figure(fig, f"{models_dir}/{model_sa}_iterative/{m}")
