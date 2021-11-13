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
import upsetplot as upset


is_plot = False

path = f"E:/YandexDisk/Work/pydnameth/datasets"
dataset = "GSE87571"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

intxn_files = {
    "Alzheimer: Roubroeks (2020)": f"{path}/lists/cpgs/neurodegenerative/Blood/Alzheimer/Roubroeks_2020.xlsx",
    "     Chronic Fatigue Syndrome: Herrera (2018)": f"{path}/lists/cpgs/neurodegenerative/Blood/Chronic Fatigue Syndrome/Herrera_2018.xlsx",
    "Parkinson: Chuang (2017)": f"{path}/lists/cpgs/neurodegenerative/Blood/Parkinson/Chuang_2017.xlsx",
    "Parkinson: Vallerga, Method 1 (2020)": f"{path}/lists/cpgs/neurodegenerative/Blood/Parkinson/Vallerga_2020_MOA.xlsx",
    "Parkinson: Vallerga, Method 2 (2020)": f"{path}/lists/cpgs/neurodegenerative/Blood/Parkinson/Vallerga_2020_MOMENT.xlsx"
}
save_files = {
    "Alzheimer: Roubroeks (2020)": "Alzheimer_Roubroeks_2020",
    "     Chronic Fatigue Syndrome: Herrera (2018)": "Chronic_Fatigue_Syndrome_Herrera_2018",
    "Parkinson: Chuang (2017)": "Parkinson_Chuang_2017",
    "Parkinson: Vallerga, Method 1 (2020)": "Parkinson_Vallerga_Method_1_2020",
    "Parkinson: Vallerga, Method 2 (2020)": "Parkinson_Vallerga_Method_2_2020"
}

features = {'Age': 'age'}

genes_name = 'beletskiy_long'
genes_fn = f"{path}/lists/genes/beletskiy/long.csv"
genes_df = pd.read_csv(genes_fn, encoding="ISO-8859-1")
genes = [x.upper() for x in genes_df['Gene Symbol'].to_list()]
# genes = ['HLA-A', 'HLA-B', 'HLA-C']

path_save = f"{path}/{platform}/{dataset}/special/001_beletskiy/{genes_name}"
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
    pval_mtx.loc[f_key, :] = -np.log10(pvals_corr)

save_df.sort_values([f"{list(features.keys())[0]}_spearman_p_value"], ascending=[True], inplace=True)
save_df.to_excel(f"{path_save}/{genes_name}.xlsx", index=True)

target_f = f"{list(features.keys())[0]}_spearman_corr_coeff"
filtered_df = save_df.loc[(save_df[target_f] >= 0.5) | (save_df[target_f] <= -0.5), :]
filtered_df.sort_values(['Gene', 'Region'], ascending=[True, True], inplace=True)
filtered_df.to_excel(f"{path_save}/{genes_name}_filtered.xlsx", index=True)
filtered_genes = list(set(filtered_df.loc[:, 'Gene'].tolist()))
np.savetxt(f"{path_save}/filtered_genes.txt", filtered_genes, delimiter="\n", fmt="%s", encoding="utf-8")

cpgs_lists = {
    "Associated with age": filtered_df.index.values,
}
for k, v in intxn_files.items():
    cpgs_df = pd.read_excel(v, header=None)
    cpgs_lists[k] = cpgs_df.iloc[:, 0].values
    tmp_df = filtered_df.loc[filtered_df.index.isin(cpgs_lists[k]), :]
    if tmp_df.shape[0] > 0:
        tmp_df.sort_values(['Gene', 'Region'], ascending=[True, True], inplace=True)
        tmp_df.to_excel(f"{path_save}/{save_files[k]}.xlsx", index=True)
    print(len(cpgs_lists[k]))
    print(len(set(cpgs_lists[k]).intersection(set(filtered_df.index.values))))

upset_df = pd.DataFrame(index=betas.columns.values)
for k, v in cpgs_lists.items():
    upset_df[k] = upset_df.index.isin(v)
upset_df = upset_df.set_index(list(cpgs_lists.keys()))
fig = upset.UpSet(upset_df, subset_size='count', show_counts=True, min_degree=2).plot()
plt.savefig(f"{path_save}/upset.png", bbox_inches='tight')
plt.savefig(f"{path_save}/upset.pdf", bbox_inches='tight')

