import pandas as pd
import statsmodels.formula.api as smf
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


x_feature = 'Age'
y_feature = 'biomarkers3_milli_Age_Control'
target_part = 'Control'

pheno = pd.read_excel(f"{path}/{platform}/{dataset}/pheno_xtd.xlsx", index_col='subject_id')

df_control = pheno.loc[pheno['Group'] == 'Control', :]
df_esrd = pheno.loc[pheno['Group'] == 'ESRD', :]

formula = f"{y_feature} ~ {x_feature}"
model = smf.ols(formula=formula, data=df_control).fit()
y_pred = model.predict(pheno)
pheno[f"{y_feature}_Acc"] = pheno[y_feature] - y_pred

pheno.to_excel(f"{path}/{platform}/{dataset}/pheno_xtd.xlsx", index=True)
pheno.to_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
