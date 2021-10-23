import os
import numpy as np
from glob import glob
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout
import plotly.express as px


model = "tabnetpl_unnhpc"
check_sum = 'd11b5f9b6efd089db42a3d5e6b375430'
output_dim = 6

cpgs_type = 'all'
counts = np.linspace(10, 200, 20, dtype=int)

metrics = ["f1_macro", "f1_weighted"]

metrics_vals_glob = {x: [] for x in metrics}
for c in counts:
    data_path = f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}/models/{model}_{cpgs_type}_{c}"
    files = glob(f"{data_path}/logs/multiruns/*/*/csv/version_0/metrics.csv")

    metrics_vals = {x: [] for x in metrics}
    for f in files:
        df = pd.read_csv(f)
        val_df = df.loc[:, ['val/' + x for x in metrics]]
        val_df.dropna(inplace=True)
        test_df = df.loc[:, ['test/' + x for x in metrics]]
        test_df.dropna(inplace=True)
        for key, value in metrics_vals.items():
            value.append(0.5 * (val_df[f"val/{key}"].max() + test_df[f"test/{key}"].max()))
    metrics_vals_df = pd.DataFrame.from_dict(metrics_vals)

    for key, value in metrics_vals_glob.items():
        value.append(metrics_vals_df[f"{key}"].max())

metrics_vals_glob_df = pd.DataFrame.from_dict(metrics_vals_glob)
metrics_vals_glob_df.to_excel(f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}/cpgs/{cpgs_type}/metrics.xlsx")

for m in metrics:
    fig = go.Figure()
    add_scatter_trace(fig, counts, metrics_vals_glob_df.loc[:, m], "", mode='lines+markers')
    add_layout(fig, f"Number of features in model", f"SHAP values", f"Best {m}")
    fig.update_layout({'colorway': px.colors.qualitative.Set1})
    save_figure(fig, f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}/cpgs/{cpgs_type}/{m}")

