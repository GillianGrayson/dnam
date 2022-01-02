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
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.histogram import add_histogram_trace
from scripts.python.routines.plot.layout import add_layout



dataset = "GSEUNN"

path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

status_col = get_column_name(dataset, 'Status').replace(' ','_')
age_col = get_column_name(dataset, 'Age').replace(' ','_')
sex_col = get_column_name(dataset, 'Sex').replace(' ','_')
status_dict = get_status_dict(dataset)
status_passed_fields = status_dict['Control'] + status_dict['Case']
sex_dict = get_sex_dict(dataset)

path_save = f"{path}/{platform}/{dataset}/special/010_immuno_part3_merge_with_age_and_sex"
pathlib.Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

continuous_vars = {}
categorical_vars = {status_col: [x.column for x in status_passed_fields], sex_col: list(sex_dict.values())}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)

immuno = pd.read_excel(f"{path}/{platform}/{dataset}/data/immuno/part3.xlsx", index_col='Sample')
ages = pd.read_excel(f"{path}/{platform}/{dataset}/data/age_L_Q_H_I.xlsx", index_col='Code')
sexes = pd.read_excel(f"{path}/{platform}/{dataset}/data/sex_L_Q_H_I.xlsx", index_col='Code')

df = pd.merge(ages, sexes, left_index=True, right_index=True)
df = pd.merge(df, immuno, left_index=True, right_index=True)
df.index.name = 'ID'
df.to_excel(f"{path}/{platform}/{dataset}/data/immuno/part3_with_age.xlsx", index=True)

no_age_codes = set(immuno.index.values) - set(ages.index.values)
if len(no_age_codes) > 0:
    print(f"Subjects with missed ages:")
    print(no_age_codes)

used_ids = pheno.loc[:, 'ID'].values
duplicate_ids = set(used_ids).intersection(set(df.index.values))
if len(duplicate_ids) > 0:
    print(f"Duplicates:")
    print(duplicate_ids)

controls = df[~df.index.str.startswith(('Q', 'H'))]

fig = go.Figure()
add_histogram_trace(fig, controls.loc[controls['Sex'] == 'M', 'Age'].values, f"Males", 5.0)
add_histogram_trace(fig, controls.loc[controls['Sex'] == 'F', 'Age'].values, f"Females", 5.0)
add_layout(fig, "Age", "Count", "")
fig.update_layout(colorway=['blue', 'red'], barmode='overlay')
save_figure(fig, f"{path_save}/figs/histogram_Age")
