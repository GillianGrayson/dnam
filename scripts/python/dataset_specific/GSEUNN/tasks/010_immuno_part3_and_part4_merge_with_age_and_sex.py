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

path_save = f"{path}/{platform}/{dataset}/special/010_immuno_part3_and_part4_merge_with_age_and_sex"
pathlib.Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

continuous_vars = {}
categorical_vars = {status_col: [x.column for x in status_passed_fields], sex_col: list(sex_dict.values())}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)

immuno3 = pd.read_excel(f"{path}/{platform}/{dataset}/data/immuno/part3.xlsx", index_col='Sample')
immuno4 = pd.read_excel(f"{path}/{platform}/{dataset}/data/immuno/part4.xlsx", index_col='Sample')
immuno4 = immuno4.loc[immuno4.index.str.match(r'(L|F|I|A|S)*', na=False), :]
coomon_samples = set(immuno3.index.values).intersection(set(immuno4.index.values))
if len(coomon_samples) > 0:
    print(f"Subjects with common ids:")
    print(coomon_samples)
immuno = pd.concat([immuno3, immuno4])
ages_sexes = pd.read_excel(f"{path}/{platform}/{dataset}/data/age_sex_L_H_A_Q_I_S.xlsx", index_col='Code')

df = pd.merge(ages_sexes, immuno, left_index=True, right_index=True)

no_age_sex_codes = set(immuno.index.values) - set(ages_sexes.index.values)
if len(no_age_sex_codes) > 0:
    print(f"Subjects with missed ages:")
    print('\n'.join(sorted(list(no_age_sex_codes))))

used_ids = pheno.loc[:, 'ID'].values
duplicate_ids = list(set(used_ids).intersection(set(df.index.values)))
if len(duplicate_ids) > 0:
    print(f"Remove duplicates:")
    print('\n'.join(sorted(duplicate_ids)))
    df.drop(duplicate_ids, inplace=True)

df.index.name = 'ID'
df.to_excel(f"{path}/{platform}/{dataset}/data/immuno/part3_part4_with_age_sex.xlsx", index=True)

controls = df[~df.index.str.startswith(('Q', 'H'))]

fig = go.Figure()
add_histogram_trace(fig, controls.loc[controls['Sex'] == 'M', 'Age'].values, f"Males ({controls.loc[controls['Sex'] == 'M', :].shape[0]})", 5.0)
add_histogram_trace(fig, controls.loc[controls['Sex'] == 'F', 'Age'].values, f"Females({controls.loc[controls['Sex'] == 'F', :].shape[0]})", 5.0)
add_layout(fig, "Age", "Count", "")
fig.update_layout(colorway=['blue', 'red'], barmode='overlay')
fig.update_layout(margin=go.layout.Margin(
    l=50,
    r=10,
    b=60,
    t=40,
    pad=0
))
save_figure(fig, f"{path_save}/figs/histogram_Age")
