import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
import os
from scripts.python.pheno.datasets.filter import filter_pheno
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.features import get_column_name, get_status_names_dict, get_status_dict, \
    get_sex_dict
from matplotlib import colors
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout


dataset = "GSEUNN"
platform = "GPL21145"
path = f"E:/YandexDisk/Work/pydnameth/datasets"

# features_type = ['immuno', 'cytokines']
# features = []
# for ft in features_type:
#     with open(f"{path}/{platform}/{dataset}/features/{ft}.txt") as f:
#         f = f.read().splitlines()
#         features.extend(f)
# features = ['Age', 'DNAmAge', 'DNAmAgeHannum', 'DNAmPhenoAge', 'DNAmGrimAge', 'PhenoAge', 'ImmunoAge'] + features

features = ['Age', 'DNAmAge', 'DNAmAgeHannum', 'DNAmPhenoAge', 'DNAmGrimAge', 'PhenoAge', 'FGF21', 'GDF15', 'CXCL9']

manifest = get_manifest(platform)
genes = ['GDF15', 'FGF21', 'CXCL9']
manifest_trgt = manifest.loc[manifest['Gene'].isin(genes), :]

status_col = get_column_name(dataset, 'Status').replace(' ','_')
age_col = get_column_name(dataset, 'Age').replace(' ','_')
sex_col = get_column_name(dataset, 'Sex').replace(' ','_')
status_dict = get_status_dict(dataset)
status_names_dict = get_status_names_dict(dataset)
sex_dict = get_sex_dict(dataset)

path_save = f"{path}/{platform}/{dataset}/special/cpgs_from_certain_genes"
if not os.path.exists(f"{path_save}/figs"):
    os.makedirs(f"{path_save}/figs")

continuous_vars = {'Age': age_col}
categorical_vars = {status_col: status_dict, sex_col: sex_dict}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")

df = pd.merge(pheno, betas, left_index=True, right_index=True)
df_ctrl = df.loc[(df[status_col] == status_dict['Control']), :]
df_case = df.loc[(df[status_col] == status_dict['Case']), :]

cpgs = list(set(manifest_trgt.index.values).intersection(set(betas.columns.values)))
manifest_trgt = manifest_trgt.loc[manifest_trgt.index.isin(cpgs), :]
cpgs = list(manifest_trgt.index.values)
cpgs_to_show = [f"{cpg}({manifest_trgt.loc[cpg, 'Gene']})" for cpg in cpgs]

for suff in ['all', 'control', 'esrd']:
    if suff == 'control':
        curr_df = df_ctrl
    elif suff == 'esrd':
        curr_df = df_case
    else:
        curr_df = df

    corr_mtx = pd.DataFrame(data=np.zeros(shape=(len(features), len(cpgs))), index=features, columns=cpgs_to_show)
    pval_mtx = pd.DataFrame(data=np.zeros(shape=(len(features), len(cpgs))), index=features, columns=cpgs_to_show)
    for f in features:
        for cpg_id, cpg in enumerate(cpgs):
            corr, pval = spearmanr(curr_df[f], curr_df[cpg])
            corr_mtx.loc[f, cpgs_to_show[cpg_id]] = corr
            pval_mtx.loc[f, cpgs_to_show[cpg_id]] = -np.log10(pval)

    mtx_to_plot = corr_mtx.to_numpy()
    cmap = plt.get_cmap("coolwarm")
    divnorm = colors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    fig, ax = plt.subplots()
    im = ax.imshow(mtx_to_plot, cmap=cmap, norm=divnorm)
    cbar = ax.figure.colorbar(im, ax=ax, location='top')
    cbar.set_label("Spearman correlation", horizontalalignment='center', fontsize=16)
    ax.set_xticks(np.arange(len(cpgs_to_show)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(cpgs_to_show)
    ax.set_yticklabels(features)
    plt.setp(ax.get_xticklabels(), rotation=90)
    for i in range(len(features)):
        for j in range(len(cpgs_to_show)):
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color="black", fontsize=5)
    fig.tight_layout()
    plt.savefig(f"{path_save}/figs/corr_mtx_{suff}.png")
    plt.savefig(f"{path_save}/figs/corr_mtx_{suff}.pdf")

    mtx_to_plot = pval_mtx.to_numpy()
    cmap = plt.get_cmap("Oranges").copy()
    cmap.set_under('#d7bfd7')
    fig, ax = plt.subplots()
    im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
    cbar = ax.figure.colorbar(im, ax=ax, location='top')
    cbar.set_label(r"$-\log_{10}(\mathrm{p-val})$", horizontalalignment='center', fontsize=16)
    ax.set_xticks(np.arange(len(cpgs_to_show)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(cpgs_to_show)
    ax.set_yticklabels(features)
    plt.setp(ax.get_xticklabels(), rotation=90)
    for i in range(len(features)):
        for j in range(len(cpgs_to_show)):
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color="black", fontsize=5)
    fig.tight_layout()
    plt.savefig(f"{path_save}/figs/pval_mtx_{suff}.png")
    plt.savefig(f"{path_save}/figs/pval_mtx_{suff}.pdf")

    for cpg_id, cpg in enumerate(cpgs):
        statistic, pvalue = mannwhitneyu(df_ctrl[cpg].values, df_case[cpg].values)
        vio = go.Figure()
        add_violin_trace(vio, df_ctrl[cpg].values, 'Control')
        add_violin_trace(vio, df_case[cpg].values, 'ESRD')
        add_layout(vio, "", cpgs_to_show[cpg_id], f"p-value: {pvalue:0.4e}")
        vio.update_layout({'colorway': ['blue', 'red']})
        save_figure(vio, f"{path_save}/figs/cpgs/vio_{cpg}")

