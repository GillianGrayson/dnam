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
import glob


is_plot = False

path = f"E:/YandexDisk/Work/pydnameth/datasets"
dataset = "GSE87571"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

intxn_files = {
    "Alzheimer: Roubroeks (2020)": f"{path}/lists/cpgs/neurodegenerative/Blood/Alzheimer/Roubroeks_2020.xlxs",
    "Chronic Fatigue Syndrome: Herrera (2018)": f"{path}/lists/cpgs/neurodegenerative/Blood/Chronic Fatigue Syndrome/Herrera_2018.xlxs",
    "Parkinson: Chuang (2017)": f"{path}/lists/cpgs/neurodegenerative/Blood/Parkinson/Chuang_2017.xlxs",
    "Parkinson: Vallerga, Method 1 (2020)": f"{path}/lists/cpgs/neurodegenerative/Blood/Parkinson/Vallerga_2020_MOA.xlxs",
    "Parkinson: Vallerga, Method 2 (2020)": f"{path}/lists/cpgs/neurodegenerative/Blood/Parkinson/Vallerga_2020_MOMENT.xlxs"
}

features = {'Age': 'age'}

genes_name = 'beletskiy_long'
genes_fn = f"{path}/lists/genes/beletskiy/long.csv"
genes_df = pd.read_csv(genes_fn, encoding="ISO-8859-1")
genes = [x.upper() for x in genes_df['Gene Symbol'].to_list()]
# genes = ['HLA-A', 'HLA-B', 'HLA-C']

path_save = f"{path}/{platform}/{dataset}/special/001_cpgs_from_certain_genes/{genes_name}"
Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

age_col = get_column_name(dataset, 'Age').replace(' ','_')
sex_col = get_column_name(dataset, 'Sex').replace(' ','_')
sex_dict = get_sex_dict(dataset)

continuous_vars = {'Age': age_col}
categorical_vars = {
    sex_col: [sex_dict[x] for x in sex_dict]
}

pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
betas = betas_drop_na(betas)

df = pd.merge(pheno, betas, left_index=True, right_index=True)

missed_genes = list(set(genes) - set(manifest['Gene'].to_list()))
np.savetxt(f"{path_save}/missed_genes.txt", missed_genes, delimiter="\n", fmt="%s", encoding="utf-8")

manifest_trgt = manifest.loc[manifest['Gene'].isin(genes), :]
cpgs = list(set(manifest_trgt.index.values).intersection(set(betas.columns.values)))
manifest_trgt = manifest_trgt.loc[manifest_trgt.index.isin(cpgs), :]
cpgs = list(manifest_trgt.index.values)
cpgs_to_show = [f"{cpg}({manifest_trgt.loc[cpg, 'Gene']}, {manifest_trgt.loc[cpg, 'UCSC_RefGene_Group']})" for cpg in cpgs]

corr_mtx = pd.DataFrame(data=np.zeros(shape=(len(features), len(cpgs))), index=list(features.keys()), columns=cpgs_to_show)
pval_mtx = pd.DataFrame(data=np.zeros(shape=(len(features), len(cpgs))), index=list(features.keys()), columns=cpgs_to_show)
cols = ['Gene', 'Region']
for f in features.keys():
    cols.append(f"{f}_spearman_corr_coeff")
    cols.append(f"{f}_spearman_p_value")
save_df = pd.DataFrame(
    data=np.zeros(shape=(len(cpgs), len(cols)), dtype=object),
    index=cpgs,
    columns=cols)
for f_key, f_val in features.items():
    for cpg_id, cpg in enumerate(cpgs):
        corr, pval = spearmanr(df[f_val], df[cpg])
        corr_mtx.loc[f_key, cpgs_to_show[cpg_id]] = corr
        pval_mtx.loc[f_key, cpgs_to_show[cpg_id]] = pval
        save_df.loc[cpgs[cpg_id], f"{f_key}_spearman_corr_coeff"] = corr
        save_df.loc[cpg, 'Gene'] = manifest_trgt.loc[cpg, 'Gene']
        save_df.loc[cpg, 'Region'] = manifest_trgt.loc[cpg, 'UCSC_RefGene_Group']
    reject, pvals_corr, alphacSidak, alphacBonf = multipletests(pval_mtx.loc[f_key, :].values, 0.05, method='fdr_bh')
    save_df.loc[:, f"{f_key}_spearman_p_value"] = pvals_corr
    pval_mtx.loc[f, :] = -np.log10(pvals_corr)

save_df.sort_values([f"{list(features.keys())[0]}_spearman_p_value"], ascending=[True], inplace=True)
save_df.to_excel(f"{path_save}/{genes_name}.xlsx", index=True)

target_f = f"{list(features.keys())[0]}_spearman_corr_coeff"
filtered_df = save_df.loc[(save_df[target_f] >= 0.5) | (save_df[target_f] <= -0.5), :]
filtered_df.sort_values(['Gene', 'Region'], ascending=[True, True], inplace=True)
save_df.to_excel(f"{path_save}/{genes_name}_filtered.xlsx", index=True)

upset_df = pd.DataFrame(index=tables_single.index)
for dataset in datasets:
    upset_df[dataset] = (tables_single[single_cols[dataset][0]] < pval_thld)
    for col in  single_cols[dataset][1::]:
        upset_df[dataset] =  upset_df[dataset] & (tables_single[col] < pval_thld)
upset_df = upset_df.set_index(datasets)
plt = upset.UpSet(upset_df, subset_size='count', show_counts=True).plot()
pyplot.savefig(f"{path_save}/single.png", bbox_inches='tight')
pyplot.savefig(f"{path_save}/single.pdf", bbox_inches='tight')

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
    plt.savefig(f"{path_save}/figs/corr_mtx.png")
    plt.savefig(f"{path_save}/figs/corr_mtx.pdf")

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
    plt.savefig(f"{path_save}/figs/pval_mtx.png")
    plt.savefig(f"{path_save}/figs/pval_mtx.pdf")

