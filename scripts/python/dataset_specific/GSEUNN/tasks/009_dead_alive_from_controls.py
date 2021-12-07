import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
import os
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from matplotlib import colors
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.box import add_box_trace
from scripts.python.routines.plot.layout import add_layout
import pathlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from functools import reduce
import plotly
from sklearn.decomposition import PCA


dataset = "GSEUNN"

path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)


features = {
    'FGF21_milli': 'FGF21',
    'GDF15_milli': 'GDF15',
    'CXCL9_milli': 'CXCL9',
}

status_col = get_column_name(dataset, 'Status').replace(' ','_')
age_col = get_column_name(dataset, 'Age').replace(' ','_')
sex_col = get_column_name(dataset, 'Sex').replace(' ','_')
status_dict = get_status_dict(dataset)
status_passed_fields = status_dict['Control'] + status_dict['Case']
sex_dict = get_sex_dict(dataset)

path_save = f"{path}/{platform}/{dataset}/special/009_dead_alive_from_controls"
pathlib.Path(f"{path_save}/figs/box").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/figs/vio").mkdir(parents=True, exist_ok=True)

continuous_vars = {v: k for k, v in features.items()}
categorical_vars = {status_col: [x.column for x in status_passed_fields], sex_col: list(sex_dict.values())}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")

df = pd.merge(pheno, betas, left_index=True, right_index=True)
df_ctrl = df.loc[(df[status_col] == 'Control'), :]

scalers = {}
for feat in features:
    scaler = StandardScaler()
    scaler.fit(np.array(df_ctrl[feat]).reshape(-1, 1))
    scalers[feat] = scaler
    df[feat] = scaler.transform(np.array(df[feat]).reshape(-1, 1))
df['HET'] = 0
for feat in features:
    df['HET'] += np.power((df[feat].values - np.mean(df[feat].values)), 2)

df_ctrl = df.loc[(df[status_col] == 'Control'), :]
df_case = df.loc[(df[status_col] == 'ESRD'), :]

df_outcome_alive = df_case.loc[(df_case['Disease_outcome'] == 'alive'), list(features.keys()) + ['HET']]
df_outcome_alive['Status'] = 'ESRD Alive'
df_outcome_dead = df_case.loc[(df_case['Disease_outcome'] == 'dead'), list(features.keys()) + ['HET']]
df_outcome_dead['Status'] = 'ESRD Dead'
df_outcome_ctrl = df_ctrl.loc[:, list(features.keys()) + ['HET']]
df_outcome_ctrl['Status'] = 'Control'

df_res = pd.concat([df_outcome_alive, df_outcome_dead, df_outcome_ctrl])
df_res.rename(columns=features, inplace=True)




fig = px.scatter_matrix(
    df_res,
    dimensions=list(features.values()),
    color=df_res["Status"]
)
fig.update_traces(diagonal_visible=False)
fig.update_traces(marker_size=3, selector=dict(type='splom'))
fig.update_layout(
    template="none",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    autosize=True,
    margin=go.layout.Margin(l=60, r=20, b=60, t=60, pad=0),
    showlegend=True
)
save_figure(fig, f"{path_save}/scatter_mtx")

pca = PCA()
components = pca.fit_transform(df_res[list(features.values())])
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}
fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(3),
    color=df_res["Status"]
)
fig.update_traces(diagonal_visible=False)
fig.update_traces(marker_size=3, selector=dict(type='splom'))
fig.update_layout(
    template="none",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    autosize=True,
    margin=go.layout.Margin(l=60, r=20, b=60, t=60, pad=0),
    showlegend=True
)
save_figure(fig, f"{path_save}/scatter_mtx_PCA")

statistic_outcome, pvalue_outcome = mannwhitneyu(df_outcome_alive['HET'].values, df_outcome_dead['HET'].values)
box_outcome = go.Figure()
add_box_trace(box_outcome, df_outcome_alive['HET'].values, f"Alive ({df_outcome_alive.shape[0]})")
add_box_trace(box_outcome, df_outcome_dead['HET'].values, f"Dead ({df_outcome_dead.shape[0]})")
add_layout(box_outcome, "", 'v', f"Mann-Whitney p-value: {pvalue_outcome:0.2e}")
box_outcome.update_layout({'colorway': ['blue', 'red']})
save_figure(box_outcome, f"{path_save}/figs/box/outcome_HET")
vio_outcome = go.Figure()
add_violin_trace(vio_outcome, df_outcome_alive['HET'].values, f"Alive ({df_outcome_alive.shape[0]})")
add_violin_trace(vio_outcome, df_outcome_dead['HET'].values, f"Dead ({df_outcome_dead.shape[0]})")
add_layout(vio_outcome, "", "v", f"Mann-Whitney p-value: {pvalue_outcome:0.2e}")
vio_outcome.update_layout({'colorway': ['blue', 'red']})
vio_outcome.update_traces(points='all', pointpos=0)
save_figure(vio_outcome, f"{path_save}/figs/vio/outcome_HET")

