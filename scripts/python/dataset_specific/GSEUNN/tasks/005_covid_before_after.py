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
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


path = f"E:/YandexDisk/Work/pydnameth/datasets"
dataset = "GSEUNN"

datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno.pkl")
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
betas = betas_drop_na(betas)
df = pd.merge(pheno, betas, left_index=True, right_index=True)
df_bef = df.loc[(df['Sample_Chronology'] == 1) & (df['ID'] != 'I64_1'), :]
df_aft = df.loc[(df['Sample_Chronology'] == 2) & (df['ID'] != 'I64_2'), :]

cpgs = betas.columns.values
result = {'CpG': cpgs}
result['Gene'] = np.zeros(len(cpgs), dtype=object)
result['Region'] = np.zeros(len(cpgs), dtype=object)
metrics = ['mw_pval']
for m in metrics:
    result[m] = np.zeros(len(cpgs))

for cpg_id, cpg in tqdm(enumerate(cpgs), desc='Regression', total=len(cpgs)):
    result['Gene'][cpg_id] = manifest.loc[cpg, 'Gene']
    result['Region'][cpg_id] = manifest.loc[cpg, 'UCSC_RefGene_Group']
    data_1 = df_bef[cpg].values
    data_2 = df_2[cpg].values
    statistic, pvalue = mannwhitneyu(df_bef[cpg].values, df_aft[cpg].values)
    result['statistic'][cpg_id] = statistic
    result['pval'][cpg_id] = pvalue

result = correct_pvalues(result, ['pval'])


statuses = ['Control', 'ESRD']
features = ['Age']

genes_name = 'beletskiy_long'
genes_fn = f"{path}/lists/genes/beletskiy/long.csv"
genes_df = pd.read_csv(genes_fn, encoding="ISO-8859-1")
genes = [x.upper() for x in genes_df['Gene Symbol'].to_list()]
# genes = ['HLA-A', 'HLA-B', 'HLA-C']

path_save = f"{path}/{platform}/{dataset}/special/001_cpgs_from_certain_genes/{genes_name}"
Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

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
missed_genes = list(set(genes) - set(manifest['Gene'].to_list()))
np.savetxt(f"{path_save}/missed_genes.txt", missed_genes, delimiter="\n", fmt="%s", encoding="utf-8")

manifest_trgt = manifest.loc[manifest['Gene'].isin(genes), :]
cpgs = list(set(manifest_trgt.index.values).intersection(set(betas.columns.values)))
manifest_trgt = manifest_trgt.loc[manifest_trgt.index.isin(cpgs), :]
cpgs = list(manifest_trgt.index.values)
cpgs_to_show = [f"{cpg}({manifest_trgt.loc[cpg, 'Gene']}, {manifest_trgt.loc[cpg, 'UCSC_RefGene_Group']})" for cpg in cpgs]

for suff in ['all', 'control', 'esrd']:
    if suff == 'control':
        curr_df = df_ctrl
    elif suff == 'esrd':
        curr_df = df_case
    else:
        curr_df = df

    corr_mtx = pd.DataFrame(data=np.zeros(shape=(len(features), len(cpgs))), index=features, columns=cpgs_to_show)
    pval_mtx = pd.DataFrame(data=np.zeros(shape=(len(features), len(cpgs))), index=features, columns=cpgs_to_show)
    cols = []
    for f in features:
        cols.append(f"{f}_spearman_corr_coeff")
        cols.append(f"{f}_spearman_p_value")
    save_df = pd.DataFrame(
        data=np.zeros(shape=(len(cpgs), 2 * len(features))),
        index=cpgs_to_show,
        columns=cols)
    save_df.index.name = 'CpG'
    for f in features:
        for cpg_id, cpg in enumerate(cpgs):
            corr, pval = spearmanr(curr_df[f], curr_df[cpg])
            corr_mtx.loc[f, cpgs_to_show[cpg_id]] = corr
            pval_mtx.loc[f, cpgs_to_show[cpg_id]] = pval
            save_df.loc[cpgs_to_show[cpg_id], f"{f}_spearman_corr_coeff"] = corr
        reject, pvals_corr, alphacSidak, alphacBonf = multipletests(pval_mtx.loc[f, :].values, 0.05, method='fdr_bh')
        save_df.loc[:, f"{f}_spearman_p_value"] = pvals_corr
        pval_mtx.loc[f, :] = -np.log10(pvals_corr)

    save_df.sort_values([f"{features[0]}_spearman_p_value"], ascending=[True], inplace=True)
    save_df.to_excel(f"{path_save}/{genes_name}_{suff}.xlsx", index=True)

    if is_plot:
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

pval_mtx = pd.DataFrame(data=np.zeros(shape=(1, len(cpgs))), index=[0], columns=cpgs_to_show)
for cpg_id, cpg in enumerate(cpgs):
    statistic, pvalue = mannwhitneyu(df_ctrl[cpg].values, df_case[cpg].values)
    pval_mtx.loc[0, cpgs_to_show[cpg_id]] = pvalue
reject, pvals_corr, alphacSidak, alphacBonf = multipletests(pval_mtx.loc[0, :].values, 0.05, method='fdr_bh')
pval_mtx.loc[0, :] = -np.log10(pvals_corr)

if is_plot:
    mtx_to_plot = pval_mtx.to_numpy()
    cmap = plt.get_cmap("Oranges").copy()
    cmap.set_under('#d7bfd7')
    fig, ax = plt.subplots()
    im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
    cbar = ax.figure.colorbar(im, ax=ax, location='top')
    cbar.set_label(r"$-\log_{10}(\mathrm{p-val})$", horizontalalignment='center', fontsize=16)
    ax.set_xticks(np.arange(len(cpgs_to_show)))
    ax.set_yticks(np.arange(1))
    ax.set_xticklabels(cpgs_to_show)
    ax.set_yticklabels(['MW'])
    plt.setp(ax.get_xticklabels(), rotation=90)
    for j in range(len(cpgs_to_show)):
        text = ax.text(j, 0, f"{mtx_to_plot[0, j]:0.2f}", ha="center", va="center", color="black", fontsize=5)
    fig.tight_layout()
    plt.savefig(f"{path_save}/figs/mw_pval_mtx_{suff}.png")
    plt.savefig(f"{path_save}/figs/mw_pval_mtx_{suff}.pdf")

if is_plot:
    for cpg_id, cpg in enumerate(cpgs):
        statistic, pvalue = mannwhitneyu(df_ctrl[cpg].values, df_case[cpg].values)
        vio = go.Figure()
        add_violin_trace(vio, df_ctrl[cpg].values, 'Control')
        add_violin_trace(vio, df_case[cpg].values, 'ESRD')
        add_layout(vio, "", cpgs_to_show[cpg_id], f"p-value: {pvalue:0.4e}")
        vio.update_layout({'colorway': ['blue', 'red']})
        Path(f"{path_save}/cpgs/{manifest_trgt.loc[cpg, 'Gene']}").mkdir(parents=True, exist_ok=True)
        save_figure(vio, f"{path_save}/figs/cpgs/{manifest_trgt.loc[cpg, 'Gene']}/{cpg}_vio")

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
            save_figure(fig, f"{path_save}/figs/cpgs/{manifest_trgt.loc[cpg, 'Gene']}/{cpg}_scatter")

