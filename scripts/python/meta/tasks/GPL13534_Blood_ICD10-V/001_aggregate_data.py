import copy
import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
from scipy.stats import kruskal
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_statuses_datasets_dict
from sklearn.feature_selection import VarianceThreshold
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.betas import betas_drop_na
import json
import pathlib
from statsmodels.stats.multitest import multipletests
from scripts.python.routines.plot.layout import add_layout, get_axis
from scripts.python.routines.plot.save import save_figure
import plotly.graph_objects as go
from tqdm import tqdm


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

statuses = [
    'Schizophrenia',
    'First episode psychosis',
    'Depression',
]
include_controls = True
datasets_control = ['GSE87571']
target_features = ['Status']

statuses_datasets_dict = get_statuses_datasets_dict()
datasets = {}
for s in statuses:
    if s in statuses_datasets_dict:
        for dataset in statuses_datasets_dict[s]:
            if dataset not in datasets:
                if include_controls:
                    datasets[dataset] = ['Control', s]
                else:
                    datasets[dataset] = [s]
            else:
                datasets[dataset].append(s)
for ds in datasets_control:
    datasets[ds] = ['Control']
datasets = dict(sorted(datasets.items()))

info = {"statuses": statuses, "datasets": datasets, 'include_controls': include_controls}

folder_name = f"GPL13534_Blood_ICD10-V"
path_save = f"{path}/meta/tasks/{folder_name}"
pathlib.Path(f"{path_save}/figs/KW").mkdir(parents=True, exist_ok=True)
with open(f"{path_save}/info.json", 'w', encoding='utf-8') as f:
    json.dump(info, f, ensure_ascii=False, indent=4)
manifest = get_manifest('GPL13534')

pheno_all = pd.DataFrame(columns=target_features + ['Dataset'])
pheno_all.index.name = 'subject_id'
for d_id, dataset in enumerate(datasets):
    print(dataset)

    platform = datasets_info.loc[dataset, 'platform']

    curr_statuses = datasets[dataset]

    status_col = get_column_name(dataset, 'Status')
    status_dict = get_status_dict(dataset)
    status_passed_fields = get_passed_fields(status_dict, curr_statuses)
    continuous_vars = {}
    categorical_vars = {status_col: [x.column for x in status_passed_fields]}
    pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno.pkl")
    pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
    betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
    betas = betas_drop_na(betas)
    df = pd.merge(pheno, betas, left_index=True, right_index=True)

    pheno = df.loc[:, [status_col]]
    status_dict_inverse = dict((x.column, x.label) for x in status_passed_fields)
    pheno[status_col].replace(status_dict_inverse, inplace=True)
    pheno.rename(columns={status_col: 'Status'}, inplace=True)
    pheno.loc[:, 'Dataset'] = dataset

    print(f"Number of subjects in {dataset}: {pheno.shape[0]}")
    print(f"Number of CpGs in {dataset}: {betas.shape[1]}")

    pheno_all = pheno_all.append(pheno, verify_integrity=True)

    cpgs = betas.columns.values
    betas = df[cpgs].T
    if d_id == 0:
        betas_all = betas
    else:
        betas_all = betas_all.merge(betas, how='inner', left_index=True, right_index=True)

betas_all = betas_all.T
betas_all.index.name = "subject_id"
betas_all = betas_all.astype('float32')
print(f"Number of total subjects: {pheno_all.shape[0]}")
print(f"Number of total CpGs: {betas_all.shape[1]}")

pheno_all, betas_all = get_pheno_betas_with_common_subjects(pheno_all, betas_all)
save_pheno_betas_to_pkl(pheno_all, betas_all, f"{path_save}")

df = pd.merge(pheno_all, betas_all, left_index=True, right_index=True)

cpgs_metrics_dict = {'CpG': betas_all.columns.values}

cpgs_metrics_dict['KW_Controls_pval'] = []
for cpg_id, cpg in enumerate(tqdm(betas_all.columns.values)):
    kw_vals = {}
    for dataset in datasets:
        vals_i = df.loc[(df['Status'] == 'Control') & (df['Dataset'] == dataset), cpg].values
        kw_vals[dataset] = vals_i
    stat, pval = kruskal(*kw_vals.values())
    cpgs_metrics_dict['KW_Controls_pval'].append(pval)
_, pvals_corr, _, _ = multipletests(cpgs_metrics_dict['KW_Controls_pval'], 0.05, method='fdr_bh')
cpgs_metrics_dict['KW_Controls_pval_fdr_bh'] = pvals_corr
_, pvals_corr, _, _ = multipletests(cpgs_metrics_dict['KW_Controls_pval'], 0.05, method='bonferroni')
cpgs_metrics_dict['KW_Controls_pval_bonferroni'] = pvals_corr

vt = VarianceThreshold(0.0)
vt.fit(betas_all)
vt_metrics = vt.variances_
vt_bool = vt.get_support()
cpgs_metrics_dict['variance'] = vt_metrics

cpgs_metrics_df = pd.DataFrame(cpgs_metrics_dict)
cpgs_metrics_df.set_index('CpG', inplace=True)
cpgs_metrics_df.sort_values(['KW_Controls_pval_fdr_bh'], ascending=[False], inplace=True)
cpgs_metrics_df.to_excel(f"{path_save}/cpgs_metrics.xlsx", index=True)

plot = cpgs_metrics_df['variance'].plot.kde(ind=np.logspace(-5, 0, 501))
plt.xlabel("Values", fontsize=15)
plt.ylabel("PDF",fontsize=15)
plt.xscale('log')
plt.grid(True)
fig = plot.get_figure()
fig.savefig(f"{path_save}/figs/{'variance'}.pdf")
fig.savefig(f"{path_save}/figs/{'variance'}.png")
plt.close()

cpgs_to_plot_df = cpgs_metrics_df.head(10)
for cpg_id, (cpg, row) in enumerate(cpgs_to_plot_df.iterrows()):
    pval = row['KW_Controls_pval_fdr_bh']
    gene = manifest.at[cpg, 'Gene']

    dist_num_bins = 30
    fig = go.Figure()
    for dataset in datasets:
        vals_i = df.loc[(df['Status'] == 'Control') & (df['Dataset'] == dataset), cpg].values
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
    add_layout(fig, "", "Methylation level", f"{gene}<br>p-value: {pval:0.2e}")
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
    save_figure(fig, f"{path_save}/figs/KW/{cpg_id:3d}_{cpg}")
