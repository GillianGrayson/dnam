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
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
betas = betas_drop_na(betas)

df = pd.merge(pheno, betas, left_index=True, right_index=True)
df.set_index('ID', inplace=True)
df_ctrl = df.loc[(df[status_col] == 'Control'), :]
df_case = df.loc[(df[status_col] == 'ESRD'), :]

path_save = f"{path}/{platform}/{dataset}/special/020_agena"
pathlib.Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

agena = pd.read_excel(f"{path}/{platform}/{dataset}/data/agena/proc.xlsx", index_col='feature')
agena = agena.T
agena.index.name = "subject_id"
agena_cpgs = list(set(agena.columns.values) - set(['Group']))

subjects_common = sorted(list(set(agena.index.values).intersection(set(df_ctrl.index.values))))
subjects_agena_only = set(agena.index.values) - set(df_ctrl.index.values)
cpgs_common = list(set(agena_cpgs).intersection(set(betas.columns.values)))

rel_diff_df = pd.DataFrame(index=subjects_common, columns=cpgs_common+['Group'])

for subject in subjects_common:
    agena_i = agena.loc[subject, agena_cpgs]
    agena_i.dropna(how='all')
    cpgs_i = sorted(list(set(agena_i.index.values).intersection(set(betas.columns.values))))
    df_i = df_ctrl.loc[subject, cpgs_i]

    rel_diff_df.at[subject, 'Group'] = agena.at[subject, 'Group']

    fig = go.Figure()
    for cpg_id, cpg in enumerate(cpgs_i):
        distrib_i = df_ctrl.loc[:, cpg].values
        fig.add_trace(
            go.Violin(
                x=[cpg] * len(distrib_i),
                y=distrib_i,
                box_visible=True,
                meanline_visible=True,
                line_color='grey',
                showlegend=False,
                opacity=1.0
            )
        )

        showlegend = False
        if cpg_id == 0:
            showlegend = True

        meth_epic = df_ctrl.at[subject, cpg]
        meth_agena = agena_i.at[cpg] * 0.01
        tmp = (meth_agena - meth_epic) / meth_epic * 100.0
        rel_diff_df.at[subject, cpg] = tmp

        fig.add_trace(
            go.Scatter(
                x=[cpg],
                y=[meth_epic],
                showlegend=showlegend,
                name="850K",
                mode="markers",
                marker=dict(
                    size=15,
                    opacity=0.7,
                    line=dict(
                        width=1
                    ),
                    color='red'
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[cpg],
                y=[meth_agena],
                showlegend=showlegend,
                name="Agena",
                mode="markers",
                marker=dict(
                    size=12,
                    opacity=0.7,
                    line=dict(
                        width=1
                    ),
                    color='blue'
                ),
            )
        )

    add_layout(fig, "", 'Methylation level', f"")
    fig.update_xaxes(tickangle=270)
    fig.update_xaxes(tickfont_size=15)
    fig.update_layout(margin=go.layout.Margin(
        l=80,
        r=20,
        b=120,
        t=50,
        pad=0
    ))
    save_figure(fig, f"{path_save}/figs/{subject}")

colors = px.colors.qualitative.Set1
groups = sorted(rel_diff_df['Group'].unique())
rel_diff_df.to_excel(f"{path_save}/rel_diff.xlsx", index=True)

fig = go.Figure()
for cpg_id, cpg in enumerate(cpgs_common):
    series_i = rel_diff_df.loc[subjects_common, cpg].dropna()
    series_i = series_i.astype('float64')
    distrib_i = series_i.values

    showlegend = False
    if cpg_id == 0:
        showlegend = True

    fig.add_trace(
        go.Violin(
            x=[cpg] * len(distrib_i),
            y=distrib_i,
            showlegend=False,
            box_visible=True,
            meanline_visible=True,
            line_color='black',
            line=dict(width=0.35),
            fillcolor='grey',
            marker=dict(color='grey', line=dict(color='black', width=0.3), opacity=0.8),
            points=False,
            bandwidth=np.ptp(distrib_i) / 25,
            opacity=0.8
        )
    )
    for g_id, g in enumerate(groups):
        series_i = rel_diff_df.loc[rel_diff_df['Group'] == g, cpg].dropna()
        series_i = series_i.astype('float64')
        distrib_i = series_i.values
        fig.add_trace(
            go.Box(
                x=[cpg] * len(distrib_i),
                name=g,
                y=distrib_i,
                boxpoints='all',
                fillcolor='rgba(255,255,255,0)',
                hoveron = 'points',
                line = {'color': 'rgba(255,255,255,0)'},
                pointpos = -2,
                showlegend = showlegend,
                marker=dict(size=4, color=colors[g_id], line=dict(color='black', width=0.3), opacity=0.6),
            )
        )

add_layout(fig, "", "Relative difference, %", f"")
fig.update_xaxes(tickangle=270)
fig.update_xaxes(tickfont_size=15)
fig.update_layout(margin=go.layout.Margin(
    l=120,
    r=20,
    b=120,
    t=50,
    pad=0
))
fig.update_layout(title_xref='paper')
fig.update_layout(legend= {'itemsizing': 'constant'})
fig.update_layout(legend_font_size=20)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)
save_figure(fig, f"{path_save}/figs/reldiff")

ololo = 1
