import numpy as np
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import plotly.express as px

data_path = f"E:/YandexDisk/Work/pydnameth/datasets/meta/classes_9/374853"
folder_path = f"{data_path}/models/xai/logs/runs/2021-10-12_23-52-32"

num_top_features = 10

statuses = {
    'Control': 0,
    'Schizophrenia': 1,
    'First episode psychosis': 2,
    'Parkinson': 3,
    'Depression': 4,
    'Intellectual disability and congenital anomalies': 5,
    'Frontotemporal dementia': 6,
    'Sporadic Creutzfeldt-Jakob disease': 7,
    'Mild cognitive impairment': 8,
    'Alzheimer': 9,
}

betas = pd.read_pickle(f"{data_path}/betas.pkl")
pheno = pd.read_pickle(f"{data_path}/pheno.pkl")
common_df = pd.merge(pheno, betas, left_index=True, right_index=True)

feat_importances = pd.read_excel(f"{folder_path}/feat_importances.xlsx", index_col='feat')

for feat_id, feat in enumerate(feat_importances.index.values[0:num_top_features]):

    curr_var = np.var(common_df.loc[:, feat].values)

    fig = go.Figure()
    for status, code_status in statuses.items():
        add_violin_trace(fig, common_df.loc[common_df['StatusFull'] == status, feat].values, status, True)
    add_layout(fig, f"variance = {curr_var:0.2e}", f"{feat}", "")
    fig.update_xaxes(showticklabels=False)
    fig.update_layout({'colorway': px.colors.qualitative.Set1})
    Path(f"{folder_path}/features/violin").mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{folder_path}/features/violin/{feat_id}_{feat}")
