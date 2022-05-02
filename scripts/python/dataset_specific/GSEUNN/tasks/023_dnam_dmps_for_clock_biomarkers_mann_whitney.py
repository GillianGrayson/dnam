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
import plotly.io as pio
pio.kaleido.scope.mathjax = None
from scipy import stats


color_ctrl = 'lime'
color_esrd = 'fuchsia'
color_f = 'red'
color_m = 'blue'

dataset = "GSEUNN"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

path_save = f"{path}/{platform}/{dataset}/special/023_dnam_dmps_for_clock_biomarkers_stat_tests"
pathlib.Path(f"{path_save}/figs/by_gene/").mkdir(parents=True, exist_ok=True)


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

genes = pd.read_excel(f"{path}/{platform}/{dataset}/special/021_ml_data/immuno/models/immuno_trn_val_lightgbm/runs/2022-04-15_19-17-12/feature_importances.xlsx").loc[:, 'feature'].values

df_res = pd.DataFrame()
for g_idx, g in enumerate(genes):
    manifest_g = manifest.loc[manifest['Gene'] == g, :]

    fig_group = go.Figure()
    fig_sex = go.Figure()
    df_group = pd.DataFrame()
    df_sex = pd.DataFrame()
    cpgs = []
    pvals_group = []
    pvals_sex = []
    pvals_age = []
    corrs_age = []
    for cpg_idx, cpg in enumerate(manifest_g.index.values):
        if cpg in df:

            df_group_g = df.loc[df['Group'].isin(['Control', 'ESRD']), ['Group', cpg]]
            df_group_g.rename(columns={cpg: 'Values'}, inplace=True)
            df_group_g['Features'] = cpg
            df_group = pd.concat([df_group, df_group_g])

            df_sex_g = df.loc[df['Sex'].isin(['F', 'M']), ['Sex', cpg]]
            df_sex_g.rename(columns={cpg: 'Values'}, inplace=True)
            df_sex_g['Features'] = cpg
            df_sex = pd.concat([df_sex, df_sex_g])

            vals_ctrl = df.loc[df['Group'] == 'Control', cpg].values
            ages_ctrl = df.loc[df['Group'] == 'Control', "Age"].values
            corr_age, pval_age = stats.pearsonr(vals_ctrl, ages_ctrl)
            pvals_age.append(pval_age)
            corrs_age.append(corr_age)

            vals_esrd = df.loc[df['Group'] == 'ESRD', cpg].values
            vals_f = df.loc[df['Sex'] == 'F', cpg].values
            vals_m = df.loc[df['Sex'] == 'M', cpg].values
            stat_group, pval_group = mannwhitneyu(vals_ctrl, vals_esrd, alternative='two-sided')
            pvals_group.append(pval_group)
            stat_sex, pval_sex = mannwhitneyu(vals_f, vals_m, alternative='two-sided')
            pvals_sex.append(pval_sex)
            cpgs.append(cpg)

    if len(cpgs) > 0:
        _, pvals_group_fdr, _, _ = multipletests(pvals_group, 0.05, method='fdr_bh')
        _, pvals_sex_fdr, _, _ = multipletests(pvals_sex, 0.05, method='fdr_bh')
        _, pvals_age_fdr, _, _ = multipletests(pvals_age, 0.05, method='fdr_bh')

        df_tmp = pd.DataFrame(
            {
                'CpG': cpgs,
                'Gene': [g] * len(cpgs),
                'Order': [g_idx] * len(cpgs),
                'pval_group': pvals_group,
                'pval_group_fdr': pvals_group_fdr,
                'pval_sex': pvals_sex,
                'pval_sex_fdr': pvals_sex_fdr,
                'pval_age': pvals_age,
                'pval_age_fdr': pvals_age_fdr,
                'corr_age': corrs_age,
            }
        )
        df_res = pd.concat([df_res, df_tmp])

        fig_group.add_trace(
            go.Violin(
                x=df_group.loc[df_group['Group'] == "Control", 'Features'],
                y=df_group.loc[df_group['Group'] == "Control", 'Values'],
                legendgroup='Control',
                scalegroup='Control',
                name='Control',
                line=dict(color='black', width=0.05),
                side='negative',
                fillcolor=color_ctrl,
                marker=dict(color=color_ctrl, line=dict(color='black', width=0.01), opacity=0.9),
            )
        )
        fig_group.add_trace(
            go.Violin(
                x=df_group.loc[df_group['Group'] == "ESRD", 'Features'],
                y=df_group.loc[df_group['Group'] == "ESRD", 'Values'],
                legendgroup='ESRD',
                scalegroup='ESRD',
                name='ESRD',
                line=dict(color='black', width=0.05),
                side='positive',
                fillcolor=color_esrd,
                marker=dict(color=color_esrd, line=dict(color='black', width=0.01), opacity=0.9),
            )
        )
        fig_group.update_traces(box_visible=True, meanline_visible=True, jitter=0.05, scalemode='width')
        add_layout(fig_group, "", 'Methylation level', f"")
        fig_group.update_xaxes(autorange=False)
        fig_group.update_layout(xaxis_range=[-0.75, len(cpgs) - 0.25])
        fig_group.update_layout(violingap=0, violingroupgap=0, violinmode='overlay')
        fig_group.update_xaxes(tickangle=270)
        fig_group.update_xaxes(tickfont_size=15)
        fig_group.update_layout(
            margin=go.layout.Margin(
                l=120,
                r=20,
                b=120,
                t=50,
                pad=0
            )
        )
        fig_group.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.15,
                xanchor="center",
                x=0.5
            )
        )
        for cpg_idx in range(len(cpgs)):
            fig_group.add_annotation(dict(font=dict(color='black', size=6),
                                    x=cpg_idx,
                                    y=1.09,
                                    showarrow=False,
                                    text=f"{pvals_group[cpg_idx]:0.1e}",
                                    textangle=0,
                                    xref="x",
                                    yref="y domain"
                                    ))
        pathlib.Path(f"{path_save}/figs/by_gene/{g_idx}_{g}").mkdir(parents=True, exist_ok=True)
        save_figure(fig_group, f"{path_save}/figs/by_gene/{g_idx}_{g}/group")

        fig_sex.add_trace(
            go.Violin(
                x=df_sex.loc[df_sex['Sex'] == "F", 'Features'],
                y=df_sex.loc[df_sex['Sex'] == "F", 'Values'],
                legendgroup='F',
                scalegroup='F',
                name='F',
                line=dict(color='black', width=0.05),
                side='negative',
                fillcolor=color_f,
                marker=dict(color=color_f, line=dict(color='black', width=0.01), opacity=0.9),
            )
        )
        fig_sex.add_trace(
            go.Violin(
                x=df_sex.loc[df_sex['Sex'] == "M", 'Features'],
                y=df_sex.loc[df_sex['Sex'] == "M", 'Values'],
                legendgroup='M',
                scalegroup='M',
                name='M',
                line=dict(color='black', width=0.05),
                side='positive',
                fillcolor=color_m,
                marker=dict(color=color_m, line=dict(color='black', width=0.01), opacity=0.9),
            )
        )
        fig_sex.update_traces(box_visible=True, meanline_visible=True, jitter=0.05, scalemode='width')
        add_layout(fig_sex, "", 'Methylation level', f"")
        fig_sex.update_xaxes(autorange=False)
        fig_sex.update_layout(xaxis_range=[-0.75, len(cpgs) - 0.25])
        fig_sex.update_layout(violingap=0, violingroupgap=0, violinmode='overlay')
        fig_sex.update_xaxes(tickangle=270)
        fig_sex.update_xaxes(tickfont_size=15)
        fig_sex.update_layout(
            margin=go.layout.Margin(
                l=120,
                r=20,
                b=120,
                t=50,
                pad=0
            )
        )
        fig_sex.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.15,
                xanchor="center",
                x=0.5
            )
        )
        for cpg_idx in range(len(cpgs)):
            fig_sex.add_annotation(dict(font=dict(color='black', size=6),
                                          x=cpg_idx,
                                          y=1.09,
                                          showarrow=False,
                                          text=f"{pvals_sex[cpg_idx]:0.1e}",
                                          textangle=0,
                                          xref="x",
                                          yref="y domain"
                                          ))
        pathlib.Path(f"{path_save}/figs/by_gene/{g_idx}_{g}").mkdir(parents=True, exist_ok=True)
        save_figure(fig_sex, f"{path_save}/figs/by_gene/{g_idx}_{g}/sex")

df_res.set_index("CpG", inplace=True)

n_top = 5
_, df_res['pvals_group_fdr_glob'], _, _ = multipletests(df_res.loc[:, 'pval_group'].values, 0.05, method='fdr_bh')
_, df_res['pvals_sex_fdr_glob'], _, _ = multipletests(df_res.loc[:, 'pval_sex'].values, 0.05, method='fdr_bh')
_, df_res['pvals_age_fdr_glob'], _, _ = multipletests(df_res.loc[:, 'pval_age'].values, 0.05, method='fdr_bh')
df_res.to_excel(f"{path_save}/res.xlsx", index=True)
df_res_top_group = df_res.sort_values(['pvals_group_fdr_glob'], ascending=[True]).head(5)
df_res_top_sex = df_res.sort_values(['pvals_sex_fdr_glob'], ascending=[True]).head(5)
df_res_top_age = df_res.sort_values(['pvals_age_fdr_glob'], ascending=[True]).head(5)

for cpg_id, (cpg, row) in enumerate(df_res_top_group.iterrows()):
    dist_num_bins = 25
    pval = row['pvals_group_fdr_glob']
    gene = manifest.at[cpg, 'Gene']
    fig = go.Figure()
    vals = df.loc[df['Group'] == 'Control', cpg].values
    fig.add_trace(
        go.Violin(
            y=vals,
            name='Control',
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
            marker=dict(line=dict(width=0.3), opacity=1),
            points='all',
            bandwidth=np.ptp(vals) / dist_num_bins,
            opacity=0.8
        )
    )
    vals = df.loc[df['Group'] == 'ESRD', cpg].values
    fig.add_trace(
        go.Violin(
            y=vals,
            name='ESRD',
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
            marker=dict(line=dict(width=0.3), opacity=1),
            points='all',
            bandwidth=np.ptp(vals) / dist_num_bins,
            opacity=0.8
        )
    )
    add_layout(fig, "", "Methylation", f"{cpg} ({gene})<br>p-value: {pval:0.2e}")
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_xaxes(tickfont_size=15)
    fig.update_layout({'colorway': [color_ctrl, color_esrd]})
    fig.update_layout(
        margin=go.layout.Margin(
            l=110,
            r=20,
            b=50,
            t=80,
            pad=0
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.25,
            xanchor="center",
            x=0.5
        )
    )
    pathlib.Path(f"{path_save}/figs/top_{n_top}/group").mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{path_save}/figs/top_{n_top}/group/{cpg_id:03d}_{cpg}")

for cpg_id, (cpg, row) in enumerate(df_res_top_sex.iterrows()):
    dist_num_bins = 25
    pval = row['pvals_sex_fdr_glob']
    gene = manifest.at[cpg, 'Gene']
    fig = go.Figure()
    vals = df.loc[df['Sex'] == 'F', cpg].values
    fig.add_trace(
        go.Violin(
            y=vals,
            name='F',
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
            marker=dict(line=dict(width=0.3), opacity=1),
            points='all',
            bandwidth=np.ptp(vals) / dist_num_bins,
            opacity=0.8
        )
    )
    vals = df.loc[df['Sex'] == 'M', cpg].values
    fig.add_trace(
        go.Violin(
            y=vals,
            name='M',
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
            marker=dict(line=dict(width=0.3), opacity=1),
            points='all',
            bandwidth=np.ptp(vals) / dist_num_bins,
            opacity=0.8
        )
    )
    add_layout(fig, "", "Methylation", f"{cpg} ({gene})<br>p-value: {pval:0.2e}")
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_xaxes(tickfont_size=15)
    fig.update_layout({'colorway': [color_f, color_m]})
    fig.update_layout(
        margin=go.layout.Margin(
            l=110,
            r=20,
            b=50,
            t=80,
            pad=0
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.25,
            xanchor="center",
            x=0.5
        )
    )
    pathlib.Path(f"{path_save}/figs/top_{n_top}/sex").mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{path_save}/figs/top_{n_top}/sex/{cpg_id:03d}_{cpg}")

for cpg_id, (cpg, row) in enumerate(df_res_top_age.iterrows()):
    dist_num_bins = 25
    pval = row['pvals_age_fdr_glob']
    corr = row['corr_age']
    gene = manifest.at[cpg, 'Gene']
    df_ctrl = df.loc[df['Group'] == 'Control', ['Age', cpg]]
    df_esrd = df.loc[df['Group'] == 'ESRD', ['Age', cpg]]

    formula = f"{cpg} ~ Age"
    model_linear = smf.ols(formula=formula, data=df_ctrl).fit()
    df_ctrl.loc[:, "Methylation acceleration"] = df_ctrl.loc[:, cpg].values - model_linear.predict(df_ctrl)
    df_esrd.loc[:, "Methylation acceleration"] = df_esrd.loc[:, cpg].values - model_linear.predict(df_esrd)
    fig = go.Figure()
    add_scatter_trace(fig, df_ctrl.loc[:, 'Age'].values, df_ctrl.loc[:, cpg].values, f"Control")
    add_scatter_trace(fig, df_ctrl.loc[:, 'Age'].values, model_linear.fittedvalues.values, "", "lines")
    add_scatter_trace(fig, df_esrd.loc[:, 'Age'].values, df_esrd.loc[:, cpg].values, f"ESRD")
    add_layout(fig, f"Age", f"{cpg} ({gene})", f"r: {corr:0.2f}    p-value: {pval:0.2e}")
    fig.update_layout({'colorway': [color_ctrl, color_ctrl, color_esrd]})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(
        title={
            'y': 1.00,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    fig.update_layout(
        margin=go.layout.Margin(
            l=120,
            r=20,
            b=80,
            t=65,
            pad=0
        )
    )
    pathlib.Path(f"{path_save}/figs/top_{n_top}/age").mkdir(parents=True, exist_ok=True)
    save_figure(fig, f"{path_save}/figs/top_{n_top}/age/{cpg_id:03d}_{cpg}_scatter")

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=df_ctrl.loc[:, "Methylation acceleration"].values,
            name=f"Control",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor=color_ctrl,
            marker=dict(color=color_ctrl, line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(df_ctrl.loc[:, "Methylation acceleration"].values) / dist_num_bins,
            opacity=0.8
        )
    )
    fig.add_trace(
        go.Violin(
            y=df_esrd.loc[:, "Methylation acceleration"].values,
            name=f"ESRD",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor=color_esrd,
            marker=dict(color=color_esrd, line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(df_esrd.loc[:, "Methylation acceleration"].values) / dist_num_bins,
            opacity=0.8
        )
    )
    add_layout(fig, "", "Methylation acceleration", f"")
    stat_01, pval_01 = mannwhitneyu(df_ctrl.loc[:, "Methylation acceleration"].values,
                                    df_esrd.loc[:, "Methylation acceleration"].values,
                                    alternative='two-sided')
    fig = add_p_value_annotation(fig, {(0, 1): pval_01})
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_layout(margin=go.layout.Margin(l=110, r=20, b=50, t=90, pad=0))
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.25,
            xanchor="center",
            x=0.5
        )
    )
    save_figure(fig, f"{path_save}/figs/top_{n_top}/age/{cpg_id:03d}_{cpg}_violin")

