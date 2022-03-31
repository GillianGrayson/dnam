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
import re

disease = "Parkinson"
data_type = "harmonized"
model_sa = 'catboost'
run_type = "trn_tst"

num_realizations = 8
n_feats = np.linspace(10, 70, 7, dtype=int)

base_dir = f"/common/home/yusipov_i/data/dnam/datasets/meta/GPL13534_Blood/{disease}"
models_dir = f"{base_dir}/{data_type}/models"

metrics = {
    "accuracy_weighted": "Accuracy",
    "f1_weighted": "F1",
    "cohen_kappa": "Cohen's kappa",
    "matthews_corrcoef": "Matthews correlation coefficient",
    "auroc_weighted": "AUROC",
}
optimized_metric = "accuracy_weighted"
optimized_part = "val"
direction = "max"

parts = ['train', 'val']

baseline_fn = f"{base_dir}/harmonized/models/baseline/{disease}_{data_type}_{run_type}_{model_sa}/runs/2022-03-30_12-10-03/metrics_val_best_0013.xlsx"
baseline_metrics_df = pd.read_excel(baseline_fn, index_col="metric")

metrics_global_df = pd.DataFrame(
    index=n_feats,
    columns=[x + f"_train" for x in list(metrics.keys())] + [x + f"_val" for x in list(metrics.keys())] + [x + f"_test" for x in list(metrics.keys())] + ['file']
)
metrics_global_df.index.name = "n_feat"
for n_feat in n_feats:
    print(n_feat)
    project_name = f'{disease}_{data_type}_{run_type}_{model_sa}_{n_feat}'
    files = glob(f"{models_dir}/{project_name}/multiruns/*/*/metrics_{optimized_part}_best_*.xlsx")

    if len(files) != num_realizations:
        print(len(files))
        print(f"Available files for {n_feat}:")
        for f in files:
            print(f)
        raise ValueError("Some files are missed!")

    metrics_local_df = pd.DataFrame(index=files, columns=[optimized_metric])
    for file in files:
        df = pd.read_excel(file, index_col="metric")
        metrics_local_df.at[file, optimized_metric] = df.at[optimized_metric, optimized_part]
    metrics_local_df.sort_values([optimized_metric], ascending=[False if direction == "max" else True], inplace=True)
    best_file = metrics_local_df.index.values[0]
    metrics_global_df.at[n_feat, 'file'] = best_file
    print(best_file)

    head, tail = os.path.split(best_file)
    search_string = 'metrics_.*_best_(\d*).xlsx'
    search_split_id = re.search(search_string, tail)
    split_id = search_split_id.group(1)
    print(split_id)

    for part in parts:
        df = pd.read_excel(f"{head}/metrics_{part}_best_{split_id}.xlsx", index_col="metric")
        for metric in metrics:
            metrics_global_df.at[n_feat, f"{metric}_{part}"] = df.at[metric, part]

Path(f"{models_dir}/iterative/{disease}_{data_type}_{run_type}_{model_sa}").mkdir(parents=True, exist_ok=True)
metrics_global_df.to_excel(f"{models_dir}/iterative/{disease}_{data_type}_{run_type}_{model_sa}/metrics.xlsx", index=True)
for p in parts:
    for m in metrics:
        fig = go.Figure()
        add_scatter_trace(fig, metrics_global_df.index.values, metrics_global_df.loc[:, f"{m}_{p}"], f"", mode='lines+markers')
        # fig.add_trace(
        #     go.Scatter(
        #         x=[n_feats[0], n_feats[-1]],
        #         y=[baseline_metrics_df.at[m, optimized_part], baseline_metrics_df.at[m, optimized_part]],
        #         showlegend=False,
        #         mode='lines',
        #         line=dict(color='black', width=3, dash='dash')
        #     )
        # )
        add_layout(fig, f"Number of features in model", f"{metrics[m]}", "")
        fig.update_layout({'colorway': ['red', 'black']})
        fig.update_layout(legend_font_size=20)
        fig.update_layout(legend={'itemsizing': 'constant'})
        fig.update_layout(
            margin=go.layout.Margin(
                l=130,
                r=20,
                b=80,
                t=20,
                pad=0
            )
        )
        save_figure(fig, f"{models_dir}/iterative/{disease}_{data_type}_{run_type}_{model_sa}/{m}_{p}")
