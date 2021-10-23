import numpy as np
from glob import glob
import pandas as pd
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.layout import add_layout
import plotly.express as px


model = "tabnetpl_unnhpc"
check_sum = 'd11b5f9b6efd089db42a3d5e6b375430'
output_dim = 6

cpgs_type = '0.01'
counts = np.linspace(10, 200, 20, dtype=int)

num_realizations = 20

metrics = ["f1_macro", "f1_weighted"]
parts = ["train", "val", "test"]
main_metric = "test/f1_weighted"

metrics_vals_glob = {f"{p}/{m}": [] for p in parts for m in metrics}
for c in counts:
    data_path = f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}/models/{model}_{cpgs_type}_{c}"
    files = glob(f"{data_path}/logs/multiruns/*/*/csv/version_0/metrics.csv")

    if len(files) != num_realizations:
        print("Available files:")
        for f in files:
            print(f)
        raise ValueError("Some files are missed!")

    metrics_vals = {f"{p}/{m}": [] for p in parts for m in metrics}
    for f in files:
        df = pd.read_csv(f)
        for p in parts:
            curr_metrics = [f"{p}/" + x for x in metrics]
            curr_df = df.loc[:, curr_metrics]
            curr_df.dropna(inplace=True)
            for m in curr_metrics:
                metrics_vals[m].append(curr_df[m].max())
    metrics_vals_df = pd.DataFrame.from_dict(metrics_vals)
    metrics_vals_df.sort_values(['test/f1_weighted'], ascending=[False], inplace=True)
    for key, value in metrics_vals_glob.items():
        value.append(metrics_vals_df[f"{key}"].values[0])

metrics_vals_glob['counts'] = list(counts)
metrics_vals_glob_df = pd.DataFrame.from_dict(metrics_vals_glob)
metrics_vals_glob_df.set_index("counts", inplace=True)
metrics_vals_glob_df.to_excel(f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}/cpgs/{cpgs_type}/metrics.xlsx", index=True)

for m in metrics:
    fig = go.Figure()
    for p in parts:
        add_scatter_trace(fig, counts, metrics_vals_glob_df.loc[:, f"{p}/{m}"], f"{p}", mode='lines+markers')
    add_layout(fig, f"Number of features in model", f"{m}", "")
    fig.update_layout({'colorway': px.colors.qualitative.Set1})
    save_figure(fig, f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}/cpgs/{cpgs_type}/{m}")

