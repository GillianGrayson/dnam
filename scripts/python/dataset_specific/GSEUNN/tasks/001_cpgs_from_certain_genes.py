import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
import statsmodels.formula.api as smf
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from matplotlib import colors
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout
from scripts.python.routines.betas import betas_drop_na
from scripts.python.routines.plot.scatter import add_scatter_trace
from pathlib import Path


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

statuses = ['Control', 'ESRD']
features = ['Age']
genes = ['HLA-A', 'HLA-B', 'HLA-C']

age_col = get_column_name(dataset, 'Age').replace(' ','_')
status_col = get_column_name(dataset, 'Status').replace(' ','_')
status_dict = get_status_dict(dataset)
status_passed_fields = get_passed_fields(status_dict, statuses)
sex_col = get_column_name(dataset, 'Sex').replace(' ','_')
sex_dict = get_sex_dict(dataset)

continuous_vars = {'Age': age_col}
categorical_vars = {
    status_col: [x.column for x in status_passed_fields],
    sex_col: [sex_dict[x] for x in sex_dict]
}

pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
betas = betas_drop_na(betas)

df = pd.merge(pheno, betas, left_index=True, right_index=True)
df_ctrl = df.loc[(df[status_col] == 'Control'), :]
df_case = df.loc[(df[status_col] == 'ESRD'), :]

manifest = get_manifest(platform)
manifest_trgt = manifest.loc[manifest['Gene'].isin(genes), :]
cpgs = list(set(manifest_trgt.index.values).intersection(set(betas.columns.values)))
manifest_trgt = manifest_trgt.loc[manifest_trgt.index.isin(cpgs), :]
cpgs = list(manifest_trgt.index.values)
cpgs_to_show = [f"{cpg}({manifest_trgt.loc[cpg, 'Gene']}, {manifest_trgt.loc[cpg, 'UCSC_RefGene_Group']})" for cpg in cpgs]

path_save = f"{path}/{platform}/{dataset}/special/cpgs_from_certain_genes/figs"
Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

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
    plt.savefig(f"{path_save}/corr_mtx_{suff}.png")
    plt.savefig(f"{path_save}/corr_mtx_{suff}.pdf")

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
    plt.savefig(f"{path_save}/pval_mtx_{suff}.png")
    plt.savefig(f"{path_save}/pval_mtx_{suff}.pdf")

for cpg_id, cpg in enumerate(cpgs):
    statistic, pvalue = mannwhitneyu(df_ctrl[cpg].values, df_case[cpg].values)
    vio = go.Figure()
    add_violin_trace(vio, df_ctrl[cpg].values, 'Control')
    add_violin_trace(vio, df_case[cpg].values, 'ESRD')
    add_layout(vio, "", cpgs_to_show[cpg_id], f"p-value: {pvalue:0.4e}")
    vio.update_layout({'colorway': ['blue', 'red']})
    Path(f"{path_save}/cpgs/{manifest_trgt.loc[cpg, 'Gene']}").mkdir(parents=True, exist_ok=True)
    save_figure(vio, f"{path_save}/cpgs/{manifest_trgt.loc[cpg, 'Gene']}/{cpg}_vio")

    for f in features:
        reg = smf.ols(formula=f"{cpg} ~ {f}", data=df_ctrl).fit()
        pvalues = dict(reg.pvalues)
        pvalue = pvalues[f]
        rsquared = reg.rsquared
        pearson_r, pearson_pval = pearsonr(df_ctrl[cpg].values, df_ctrl[f].values)
        spearman_r, spearman_pval = spearmanr(df_ctrl[cpg].values, df_ctrl[f].values)

        fig = go.Figure()
        add_scatter_trace(fig, df_ctrl[f].values, df_ctrl[cpg].values, 'Control')
        add_scatter_trace(fig, df_ctrl[f].values, reg.fittedvalues.values, "", "lines")
        add_scatter_trace(fig, df_case[f].values, df_case[cpg].values, 'ESRD')
        title = fr"$R^2={{{rsquared:0.3f}}} \
        \quad \text{{{'Spearman r'}}}={{{spearman_r:0.3f}}} \
        \quad \text{{{'Spearman p-val'}}}={{{spearman_pval:0.3e}}}$"
        add_layout(fig, f"{f}", cpgs_to_show[cpg_id], title=title)
        fig.update_layout({'colorway': ['blue', 'blue', "red"]})
        save_figure(fig, f"{path_save}/cpgs/{manifest_trgt.loc[cpg, 'Gene']}/{cpg}_scatter")

