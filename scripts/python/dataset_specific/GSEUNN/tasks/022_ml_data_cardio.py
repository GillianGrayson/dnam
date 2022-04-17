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

path_save = f"{path}/{platform}/{dataset}/special/022_ml_data_cardio"
pathlib.Path(f"{path_save}/snp_ecg").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/snp_dnam").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/snp_ecg_dnam").mkdir(parents=True, exist_ok=True)

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
betas_features = betas.columns.values
df = pd.merge(pheno, betas, left_index=True, right_index=True)
df.set_index('ID', inplace=True)

snp = pd.read_excel(f"{path}/{platform}/{dataset}/data/snp_cardio/snp_df.xlsx", index_col='index')
snp = snp.loc[:, ["Cardiorisk", "Age", "Sex"]]
ecg = pd.read_excel(f"{path}/{platform}/{dataset}/data/ecg/ecg_df.xlsx", index_col='index')
ecg_features = pd.read_excel(f"{path}/{platform}/{dataset}/data/ecg/features_df.xlsx").loc[:, 'features'].values
ecg = ecg.loc[:, ecg_features]

subjs_snp_ecg = sorted(list(set(snp.index.values).intersection(set(ecg.index.values))))
snp_ecg = pd.merge(snp, ecg, left_index=True, right_index=True)
subjs_snp_dnam = sorted(list(set(snp.index.values).intersection(set(df.index.values))))
snp_dnam = pd.merge(snp, df.loc[:, betas_features], left_index=True, right_index=True)

snp_ecg.to_pickle(f"{path_save}/snp_ecg/data.pkl")
pd.DataFrame({'features': ecg_features}).to_excel(f"{path_save}/snp_ecg/features.xlsx", index=False)
snp_dnam.to_pickle(f"{path_save}/snp_dnam/data.pkl")
pd.DataFrame({'features': betas_features}).to_excel(f"{path_save}/snp_dnam/features.xlsx", index=False)
