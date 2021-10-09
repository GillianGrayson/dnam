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
from scripts.python.routines.plot.layout import add_layout


dataset = "GSEUNN"

path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)


features = {
    'FGF21_milli': 'FGF21',
    'GDF15_milli': 'GDF15',
    'CXCL9_milli': 'CXCL9',
    '3biomarkers_milli_Age_Control': 'EstimatedAge'
}

status_col = get_column_name(dataset, 'Status').replace(' ','_')
age_col = get_column_name(dataset, 'Age').replace(' ','_')
sex_col = get_column_name(dataset, 'Sex').replace(' ','_')
status_dict = get_status_dict(dataset)
status_passed_fields = status_dict['Control'] + status_dict['Case']
sex_dict = get_sex_dict(dataset)

path_save = f"{path}/{platform}/{dataset}/special/002_disease_grous_statistic"
if not os.path.exists(f"{path_save}/figs"):
    os.makedirs(f"{path_save}/figs")

continuous_vars = {v: k for k, v in features.items()}
categorical_vars = {status_col: [x.column for x in status_passed_fields], sex_col: list(sex_dict.values())}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")

df = pd.merge(pheno, betas, left_index=True, right_index=True)

df_cause_inf = df.loc[(df['Cause_of_the_disease'] == 'Inflammatory'), :]
df_cause_gen = df.loc[(df['Cause_of_the_disease'] == 'genetic'), :]

df_outcome_alive = df.loc[(df['Disease_outcome'] == 'alive'), :]
df_outcome_dead = df.loc[(df['Disease_outcome'] == 'dead'), :]

for feat_column, feat_show in features.items():

    statistic_cause, pvalue_cause = mannwhitneyu(df_cause_inf[feat_column].values, df_cause_gen[feat_column].values)
    vio_cause = go.Figure()
    add_violin_trace(vio_cause, df_cause_inf[feat_column].values, f"Inflammatory ({df_cause_inf.shape[0]})")
    add_violin_trace(vio_cause, df_cause_gen[feat_column].values, f"Genetic ({df_cause_gen.shape[0]})")
    add_layout(vio_cause, "", feat_show, f"Mann-Whitney p-value: {pvalue_cause:0.2e}")
    vio_cause.update_layout({'colorway': ['blue', 'red']})
    vio_cause.update_traces(points='all', pointpos=0)
    save_figure(vio_cause, f"{path_save}/figs/cause_vio_{feat_show}")

    statistic_outcome, pvalue_outcome = mannwhitneyu(df_outcome_alive[feat_column].values, df_outcome_dead[feat_column].values)
    vio_outcome = go.Figure()
    add_violin_trace(vio_outcome, df_outcome_alive[feat_column].values, f"Alive ({df_outcome_alive.shape[0]})")
    add_violin_trace(vio_outcome, df_outcome_dead[feat_column].values, f"Dead ({df_outcome_dead.shape[0]})")
    add_layout(vio_outcome, "", feat_show, f"Mann-Whitney p-value: {pvalue_outcome:0.2e}")
    vio_outcome.update_layout({'colorway': ['blue', 'red']})
    vio_outcome.update_traces(points='all', pointpos=0)
    save_figure(vio_outcome, f"{path_save}/figs/outcome_vio_{feat_show}")
