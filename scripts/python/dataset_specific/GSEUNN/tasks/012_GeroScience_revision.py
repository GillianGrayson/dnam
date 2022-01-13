import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
import os
import pickle
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from scripts.python.routines.plot.scatter import add_scatter_trace
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
from functools import reduce
import plotly
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.histogram import add_histogram_trace
from scripts.python.routines.plot.layout import add_layout
from scripts.python.routines.plot.p_value import add_p_value_annotation
from scripts.python.EWAS.routines.correction import correct_pvalues
import plotly.express as px


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
continuous_vars = {}
categorical_vars = {status_col: [x.column for x in status_passed_fields], sex_col: list(sex_dict.values())}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
ctrl = pheno.loc[pheno['Group'] == 'Control']
esrd = pheno.loc[pheno['Group'] == 'ESRD']

ctrl_color = 'lime'
esrd_color = 'fuchsia'
dist_num_bins = 25

path_save = f"{path}/{platform}/{dataset}/special/012_GeroScience_revision"
pathlib.Path(f"{path_save}/SupplementaryFigure2").mkdir(parents=True, exist_ok=True)

# Supplementary Figure 2 ===============================================================================================
result_dict = {'feature': ['CD4T', 'CD8naive', 'CD8pCD28nCD45RAn', 'Gran', 'Mono', 'NK', 'PlasmaBlast']}
result_dict['pval_mv'] = np.zeros(len(result_dict['feature']))
for f_id, f in enumerate(result_dict['feature']):
    values_ctrl = ctrl.loc[:, f].values
    values_esrd = esrd.loc[:, f].values
    stat, pval = mannwhitneyu(values_ctrl, values_esrd, alternative='two-sided')
    result_dict['pval_mv'][f_id] = pval

result_dict = correct_pvalues(result_dict, ['pval_mv'])
result_df = pd.DataFrame(result_dict)
result_df.set_index('feature', inplace=True)
result_df.sort_values(['pval_mv'], ascending=[True], inplace=True)
result_df.to_excel(f"{path_save}/SupplementaryFigure2/result.xlsx", index=True)

for f_id, f in enumerate(result_dict['feature']):
    values_ctrl = ctrl.loc[:, f].values
    values_esrd = esrd.loc[:, f].values
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=values_ctrl,
            name=f"Control",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor=ctrl_color,
            marker = dict(color=ctrl_color, line=dict(color='black',width=0.3), opacity=0.8),
            points='all',
            bandwidth = np.ptp(values_ctrl) / dist_num_bins,
            opacity=0.8
        )
    )
    fig.add_trace(
        go.Violin(
            y=values_esrd,
            name=f"ESRD",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor=esrd_color,
            marker=dict(color=esrd_color, line=dict(color='black',width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(values_ctrl) / dist_num_bins,
            opacity=0.8
        )
    )
    add_layout(fig, "", f"{f}", f"p-value: {result_df.at[f, 'pval_mv_fdr_bh']:0.2e}")
    fig.update_layout({'colorway': ['lime', 'fuchsia']})
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_layout(
        margin=go.layout.Margin(
            l=110,
            r=20,
            b=50,
            t=90,
            pad=0
        )
    )
    #fig.update_xaxes(showgrid=True)
    #fig.update_yaxes(showgrid=True)
    fig.update_layout(legend_y=1.01)
    save_figure(fig, f"{path_save}/SupplementaryFigure2/{f}")

ololo = 1