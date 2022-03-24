import numpy as np
from glob import glob
import pandas as pd
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.layout import add_layout
import plotly.express as px


check_sum = '121da597d6d3fe7b3b1b22a0ddc26e61'
seed = 2

cpgs_list_origin = "24829"
cpgs_from_model = 'tabnetpl'
cpgs_from_run = 'average'
cpgs_from_variance = 'all'
counts = np.linspace(10, 500, 50, dtype=int)

num_realizations = 12

metrics = ["f1_macro", "f1_weighted"]
parts = ["test"] # "train", "val"
main_metric = "test/f1_weighted"

metrics_vals_glob = {f"{p}/{m}": [] for p in parts for m in metrics}
for c in counts:
    print(c)
    project_name = f'{cpgs_from_model}_unnhpc_{cpgs_from_run}_{cpgs_from_variance}_{c}'
    data_path = f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}/models/{cpgs_list_origin}/{project_name}"
    files = glob(f"{data_path}/logs/multiruns/*/*/csv/version_0/metrics.csv")

    if len(files) != num_realizations:
        print(len(files))
        print(f"Available files for {c}:")
        for f in files:
            print(f)
        raise ValueError("Some files are missed!")

    metrics_vals = {f"{p}/{m}": [] for p in parts for m in metrics}
    for f in files:
        print(f)
        df = pd.read_csv(f)
        for p in parts:
            curr_metrics = [f"{p}/" + x for x in metrics]
            set_cols = set(curr_metrics).intersection(set(df.columns.values))
            if len(set_cols) != len(curr_metrics):
                print(set_cols)
                raise ValueError("Missed columns in csv logger")
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
metrics_vals_glob_df.to_excel(f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}/cpgs/{cpgs_list_origin}/{cpgs_from_model}/{cpgs_from_run}/{cpgs_from_variance}/metrics.xlsx", index=True)

for m in metrics:
    fig = go.Figure()
    for p in parts:
        add_scatter_trace(fig, counts, metrics_vals_glob_df.loc[:, f"{p}/{m}"], f"{p}", mode='lines+markers')
    add_layout(fig, f"Number of features in model", f"{m}", "")
    fig.update_layout({'colorway': px.colors.qualitative.Set1})
    save_figure(fig, f"/home/yusipov_i/data/dnam/datasets/meta/{check_sum}/cpgs/{cpgs_list_origin}/{cpgs_from_model}/{cpgs_from_run}/{cpgs_from_variance}/{m}")
