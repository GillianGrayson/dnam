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

path_save = f"{path}/{platform}/{dataset}/special/007_heterogeneity_dead_alive"
pathlib.Path(f"{path_save}/figs/box").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/figs/vio").mkdir(parents=True, exist_ok=True)

continuous_vars = {v: k for k, v in features.items()}
categorical_vars = {status_col: [x.column for x in status_passed_fields], sex_col: list(sex_dict.values())}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")

df = pd.merge(pheno, betas, left_index=True, right_index=True)

df_case = df.loc[(df[status_col] == 'ESRD'), :]

for feat in features:
    df_case[feat] = StandardScaler().fit_transform(np.array(df_case[feat]).reshape(-1, 1))

df_case['HET'] = 0

for feat in features:
    tmp_1 = df_case[feat].values - np.mean(df_case[feat].values)
    tmp_2 = np.power(tmp_1, 2)
    df_case['HET'] += np.power((df_case[feat].values - np.mean(df_case[feat].values)), 2)

df_outcome_alive = df_case.loc[(df_case['Disease_outcome'] == 'alive'), :]
df_outcome_dead = df_case.loc[(df_case['Disease_outcome'] == 'dead'), :]

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
save_figure(vio_outcome, f"{path_save}/figs/vio/outcome_{'HET'}")
