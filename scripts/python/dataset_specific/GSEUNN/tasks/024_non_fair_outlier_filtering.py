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
from sklearn.metrics import mean_absolute_error


dataset = "GSEUNN"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

path_save = f"{path}/{platform}/{dataset}/special/024_non_fair_outlier_filtering"

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
pheno.set_index('ID', inplace=True)
df = pheno.copy()

part_3_4 = pd.read_excel(f"{path}/{platform}/{dataset}/data/immuno/part3_part4_with_age_sex.xlsx", index_col='ID')
part_3_4 = part_3_4[~part_3_4.index.str.startswith(('Q', 'H'))]
part_3_4['Group'] = 'Control'
part_3_4['Source'] = 2

pheno = pheno.append(part_3_4, verify_integrity=True)
pheno = pheno.loc[(pheno['Group'] == 'Control'), :]

ctrl_paper_df = pd.read_excel(f"{path}/{platform}/{dataset}/special/011_immuno_part3_and_part4_check_clocks/part3_part4_filtered_with_age_sex_16.xlsx", index_col='ID')
ctrl_paper_indexes = df.loc[df['Group'] == 'Control', :].index.values
ctrl_paper_indexes = set(ctrl_paper_indexes).union(ctrl_paper_df.index.values)

ctrl_non_paper_indexes = set(pheno.index.values) - ctrl_paper_indexes

immuno_data = pheno.loc[pheno['Group'] == 'Control']
immuno_data.index.name = 'index'
immuno_data_paper = immuno_data.loc[immuno_data.index.isin(ctrl_paper_indexes), :]
immuno_data_not_paper = immuno_data.loc[immuno_data.index.isin(ctrl_non_paper_indexes), :]

model = pickle.load(open(f"{path}/{platform}/{dataset}/special/021_ml_data/immuno/models/immuno_trn_val_elastic_net/runs/2022-05-17_17-03-47/elastic_net_best_0017.pkl", 'rb'))
features_df = pd.read_excel(f"{path}/{platform}/{dataset}/special/021_ml_data/immuno/features.xlsx")
features = features_df.loc[:, "features"].values

X_paper = immuno_data_paper.loc[:, features].values
y_paper = immuno_data_paper.loc[:, "Age"].values
y_paper_pred = model.predict(X_paper)
mae_paper = mean_absolute_error(y_paper, y_paper_pred)

X_non_paper = immuno_data_not_paper.loc[:, features].values
y_non_paper = immuno_data_not_paper.loc[:, "Age"].values
y_non_paper_pred = model.predict(X_non_paper)
mae_non_paper = mean_absolute_error(y_non_paper, y_non_paper_pred)
y_non_paper_diff = y_non_paper_pred - y_non_paper
y_non_paper_diff_abs = np.abs(y_non_paper_diff)

immuno_data_not_paper['y_non_paper_pred'] = y_non_paper_pred
immuno_data_not_paper['y_non_paper_diff'] = y_non_paper_diff
immuno_data_not_paper['y_non_paper_diff_abs'] = y_non_paper_diff_abs

immuno_data_not_paper.to_excel(f"{path_save}/immuno_data_not_paper.xlsx", index=True)
