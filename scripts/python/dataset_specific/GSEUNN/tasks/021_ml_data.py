import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.python.routines.betas import betas_drop_na
import pickle
import random
import plotly.express as px
import copy
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scripts.python.pheno.datasets.filter import filter_pheno
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from scripts.python.routines.plot.scatter import add_scatter_trace
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
import pathlib
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout, get_axis
from scripts.python.routines.plot.p_value import add_p_value_annotation
from statsmodels.stats.multitest import multipletests


dataset = "GSEUNN"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

path_save = f"{path}/{platform}/{dataset}/special/021_ml_data"
pathlib.Path(f"{path_save}/immuno").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/agena_immuno").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/cogn_immuno").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/agena_cogn_immuno").mkdir(parents=True, exist_ok=True)

status_col = get_column_name(dataset, 'Status').replace(' ','_')
age_col = get_column_name(dataset, 'Age').replace(' ','_')
sex_col = get_column_name(dataset, 'Sex').replace(' ','_')
status_dict = get_status_dict(dataset)
status_passed_fields = status_dict['Control'] + status_dict['Case']
sex_dict = get_sex_dict(dataset)
continuous_vars = {}
categorical_vars = {status_col: [x.column for x in status_passed_fields], sex_col: list(sex_dict.values())}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
betas = betas_drop_na(betas)
df = pd.merge(pheno, betas, left_index=True, right_index=True)
df.set_index('ID', inplace=True)
df['Source'] = 1

part_3_4 = pd.read_excel(f"{path}/{platform}/{dataset}/data/immuno/part3_part4_with_age_sex.xlsx", index_col='ID')
part_3_4 = part_3_4[~part_3_4.index.str.startswith(('Q', 'H'))]
part_3_4['Group'] = 'Control'
part_3_4['Source'] = 2

pheno.set_index('ID', inplace=True)
pheno = pheno.append(part_3_4, verify_integrity=True)
pheno = pheno.loc[(pheno['Group'] == 'Control'), :]

agena = pd.read_excel(f"{path}/{platform}/{dataset}/data/agena/35.xlsx", index_col='CpG')
agena = agena.T
agena.index.name = "subject_id"
agena_cpgs = list(set(agena.columns.values))
agena.loc[:, agena_cpgs] *= 0.01
subjects_common_agena = sorted(list(set(agena.index.values).intersection(set(df.index.values))))
subjects_agena_only = set(agena.index.values) - set(df.index.values)
cpgs_common_agena = sorted(list(set(agena_cpgs).intersection(set(betas.columns.values))))

cogn = pd.read_excel(f"{path}/{platform}/{dataset}/data/cognitive/data.xlsx", index_col='subject_id')
cogn = cogn[~cogn.index.str.startswith(('Q', 'H'))]
subjects_common_cogn_df = sorted(list(set(cogn.index.values).intersection(set(df.index.values))))
subjects_common_cogn_immuno = sorted(list(set(cogn.index.values).intersection(set(pheno.index.values))))
subjects_cogn_minus_df = sorted(list(set(cogn.index.values) - set(df.index.values)))
subjects_cogn_minus_pheno = sorted(list(set(cogn.index.values) - set(pheno.index.values)))
subjects_pheno_minus_cogn = sorted(list(set(pheno.index.values) - set(cogn.index.values)))

immuno_data = pheno.loc[pheno['Group'] == 'Control']
agena_immuno_data = pd.merge(pheno.loc[pheno.index.isin(subjects_common_agena), :], agena, left_index=True, right_index=True)
cogn_immuno_data = pd.merge(pheno.loc[pheno.index.isin(subjects_common_cogn_immuno), :], cogn, left_index=True, right_index=True)
agena_cogn_immuno_data = pd.merge(cogn_immuno_data, agena, left_index=True, right_index=True)

immuno_data.to_excel(f"{path_save}/immuno/data.xlsx", index=True)
agena_immuno_data.to_excel(f"{path_save}/agena_immuno/data.xlsx", index=True)
cogn_immuno_data.to_excel(f"{path_save}/cogn_immuno/data.xlsx", index=True)
agena_cogn_immuno_data.to_excel(f"{path_save}/agena_cogn_immuno/data.xlsx", index=True)
