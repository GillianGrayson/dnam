import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
import pathlib
from scripts.python.meta.tasks.GPL13534_Blood_ICD10_V.routines import KW_Control
from tqdm import tqdm
import numpy as np
from scripts.python.routines.plot.layout import add_layout, get_axis
from scripts.python.routines.plot.save import save_figure
import plotly.graph_objects as go


thld_above = 0.5
thld_below = 0.05

path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
manifest = get_manifest('GPL13534')
dataset_statuses = {
    'GSE152027': ['Control', 'Schizophrenia'],
    'GSE84727': ['Control', 'Schizophrenia'],
    'GSE80417': ['Control', 'Schizophrenia'],
    'GSE116379': ['Control', 'Schizophrenia'],
    'GSE41169': ['Control', 'Schizophrenia'],
    'GSE116378': ['Control', 'Schizophrenia'],
}
datasets_train_val = ['GSE152027', 'GSE84727', 'GSE80417']

task_name = f"GPL13534_Blood_Schizo_Control"
path_wd = f"{path}/meta/tasks/{task_name}"
pathlib.Path(f"{path_wd}/tasks/011_harmonization_effect_on_controls/one_by_one").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_wd}/tasks/011_harmonization_effect_on_controls/all_in_one").mkdir(parents=True, exist_ok=True)

cpgs_origin = pd.read_excel(f"{path_wd}/origin/KW/cpgs_metrics.xlsx", index_col="CpG")
cpgs_one_by_one = pd.read_excel(f"{path_wd}/one_by_one/KW/cpgs_metrics.xlsx", index_col="CpG")
cpgs_all_in_one = pd.read_excel(f"{path_wd}/all_in_one/KW/cpgs_metrics.xlsx", index_col="CpG")

cpgs_info = cpgs_origin.merge(cpgs_one_by_one, left_index=True, right_index=True, suffixes=('_origin', '_one_by_one'))
cpgs_info = cpgs_info.merge(cpgs_all_in_one, left_index=True, right_index=True, suffixes=('', '_all_in_one'))
cpgs_info.rename(columns={x: x + '_all_in_one' for x in cpgs_all_in_one.columns.values}, inplace=True)
cpgs_info['log_diff_one_by_one'] = np.log10(cpgs_info.loc[:, 'KW_Controls_pval_fdr_bh_one_by_one'].values) - np.log10(cpgs_info.loc[:, 'KW_Controls_pval_fdr_bh_origin'].values)
cpgs_info['log_diff_all_in_one'] = np.log10(cpgs_info.loc[:, 'KW_Controls_pval_fdr_bh_all_in_one'].values) - np.log10(cpgs_info.loc[:, 'KW_Controls_pval_fdr_bh_origin'].values)

cpgs_changed_one_by_one = cpgs_info.loc[(cpgs_info['KW_Controls_pval_fdr_bh_one_by_one'] > thld_above) & (cpgs_info['KW_Controls_pval_fdr_bh_origin'] < thld_below), :]
cpgs_changed_one_by_one.sort_values(['log_diff_one_by_one'], ascending=[False], inplace=True)
cpgs_changed_all_in_one = cpgs_info.loc[(cpgs_info['KW_Controls_pval_fdr_bh_all_in_one'] > thld_above) & (cpgs_info['KW_Controls_pval_fdr_bh_origin'] < thld_below), :]
cpgs_changed_all_in_one.sort_values(['log_diff_all_in_one'], ascending=[False], inplace=True)
cpgs_changed_one_by_one.to_excel(f"{path_wd}/tasks/011_harmonization_effect_on_controls/cpgs_changed_one_by_one_{thld_above}_{thld_below}.xlsx", index=True)
cpgs_changed_all_in_one.to_excel(f"{path_wd}/tasks/011_harmonization_effect_on_controls/cpgs_changed_all_in_one_{thld_above}_{thld_below}.xlsx", index=True)

pheno = pd.read_pickle(f"{path_wd}/origin/pheno_train_val.pkl")
mvals = pd.read_pickle(f"{path_wd}/origin/mvalsT_train_val.pkl")
mvals = mvals.T
df_origin = pd.merge(pheno, mvals, left_index=True, right_index=True)

pheno = pd.read_pickle(f"{path_wd}/one_by_one/pheno_train_val.pkl")
mvals = pd.read_pickle(f"{path_wd}/one_by_one/mvals_train_val.pkl")
df_one_by_one = pd.merge(pheno, mvals, left_index=True, right_index=True)

pheno = pd.read_pickle(f"{path_wd}/all_in_one/pheno_train_val.pkl")
mvals = pd.read_pickle(f"{path_wd}/all_in_one/mvals_train_val.pkl")
df_all_in_one = pd.merge(pheno, mvals, left_index=True, right_index=True)

cpgs_to_plot_df = cpgs_changed_one_by_one.head(20)
for cpg_id, (cpg, row) in enumerate(cpgs_to_plot_df.iterrows()):
    dist_num_bins = 30

    pval = row['KW_Controls_pval_fdr_bh_origin']
    gene = manifest.at[cpg, 'Gene']
    fig = go.Figure()
    for dataset in datasets_train_val:
        vals_i = df_origin.loc[(df_origin['Status'] == 'Control') & (df_origin['Dataset'] == dataset), cpg].values
        fig.add_trace(
            go.Violin(
                y=vals_i,
                name=dataset,
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
                marker=dict(line=dict(width=0.3), opacity=0.8),
                points='all',
                bandwidth=np.ptp(vals_i) / dist_num_bins,
                opacity=0.8
            )
        )
    add_layout(fig, "", "M value", f"{cpg} ({gene})<br>p-value: {pval:0.2e}")
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_xaxes(tickfont_size=15)
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
    save_figure(fig, f"{path_wd}/tasks/011_harmonization_effect_on_controls/one_by_one/{cpg_id:3d}_{cpg}_origin")

    pval = row['KW_Controls_pval_fdr_bh_one_by_one']
    gene = manifest.at[cpg, 'Gene']
    fig = go.Figure()
    for dataset in datasets_train_val:
        vals_i = df_one_by_one.loc[(df_one_by_one['Status'] == 'Control') & (df_one_by_one['Dataset'] == dataset), cpg].values
        fig.add_trace(
            go.Violin(
                y=vals_i,
                name=dataset,
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
                marker=dict(line=dict(width=0.3), opacity=0.8),
                points='all',
                bandwidth=np.ptp(vals_i) / dist_num_bins,
                opacity=0.8
            )
        )
    add_layout(fig, "", "M value", f"{cpg} ({gene})<br>p-value: {pval:0.2e}")
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_xaxes(tickfont_size=15)
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
    save_figure(fig, f"{path_wd}/tasks/011_harmonization_effect_on_controls/one_by_one/{cpg_id:3d}_{cpg}_one_by_one")


cpgs_to_plot_df = cpgs_changed_all_in_one.head(20)
for cpg_id, (cpg, row) in enumerate(cpgs_to_plot_df.iterrows()):
    dist_num_bins = 30

    pval = row['KW_Controls_pval_fdr_bh_origin']
    gene = manifest.at[cpg, 'Gene']
    fig = go.Figure()
    for dataset in datasets_train_val:
        vals_i = df_origin.loc[(df_origin['Status'] == 'Control') & (df_origin['Dataset'] == dataset), cpg].values
        fig.add_trace(
            go.Violin(
                y=vals_i,
                name=dataset,
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
                marker=dict(line=dict(width=0.3), opacity=0.8),
                points='all',
                bandwidth=np.ptp(vals_i) / dist_num_bins,
                opacity=0.8
            )
        )
    add_layout(fig, "", "M value", f"{cpg} ({gene})<br>p-value: {pval:0.2e}")
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_xaxes(tickfont_size=15)
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
    save_figure(fig, f"{path_wd}/tasks/011_harmonization_effect_on_controls/all_in_one/{cpg_id:3d}_{cpg}_origin")

    pval = row['KW_Controls_pval_fdr_bh_all_in_one']
    gene = manifest.at[cpg, 'Gene']
    fig = go.Figure()
    for dataset in datasets_train_val:
        vals_i = df_all_in_one.loc[(df_all_in_one['Status'] == 'Control') & (df_all_in_one['Dataset'] == dataset), cpg].values
        fig.add_trace(
            go.Violin(
                y=vals_i,
                name=dataset,
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
                marker=dict(line=dict(width=0.3), opacity=0.8),
                points='all',
                bandwidth=np.ptp(vals_i) / dist_num_bins,
                opacity=0.8
            )
        )
    add_layout(fig, "", "M value", f"{cpg} ({gene})<br>p-value: {pval:0.2e}")
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_xaxes(tickfont_size=15)
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
    save_figure(fig, f"{path_wd}/tasks/011_harmonization_effect_on_controls/all_in_one/{cpg_id:3d}_{cpg}_all_in_one")

