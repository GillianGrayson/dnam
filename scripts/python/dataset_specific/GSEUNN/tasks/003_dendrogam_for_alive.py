import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
import os
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scipy.stats import spearmanr
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from matplotlib import colors
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.box import add_box_trace
from scripts.python.routines.plot.layout import add_layout
import pathlib
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff


dataset = "GSEUNN"

path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

path_save = f"{path}/{platform}/{dataset}/special/003_dendrogam_for_alive"
if not os.path.exists(f"{path_save}/figs"):
    pathlib.Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

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

continuous_vars = {v: k for k, v in features.items()}
categorical_vars = {status_col: [x.column for x in status_passed_fields], sex_col: list(sex_dict.values())}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
df = pd.merge(pheno, betas, left_index=True, right_index=True)

df_outcome_alive = df.loc[(df['Disease_outcome'] == 'alive'), ["ID"] + list(features.keys())]
df_outcome_alive.set_index("ID", inplace=True)
df_outcome_alive.rename(columns=features, inplace=True)
# Normalization
df_outcome_alive = (df_outcome_alive - df_outcome_alive.mean()) / df_outcome_alive.std()

Z = linkage(df_outcome_alive, 'ward')
dendrogram(Z, leaf_rotation=90, leaf_font_size=6, labels=df_outcome_alive.index)
plt.savefig(f"{path_save}/figs/dendrogram.png")
plt.savefig(f"{path_save}/figs/dendrogram.pdf")
