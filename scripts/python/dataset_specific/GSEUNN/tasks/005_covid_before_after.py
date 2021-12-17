import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.box import add_box_trace
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout
from scripts.python.routines.betas import betas_drop_na
from pathlib import Path
from tqdm import tqdm
from scripts.python.EWAS.routines.correction import correct_pvalues
from scripts.python.routines.manifest import get_genes_list


path = f"E:/YandexDisk/Work/pydnameth/datasets"
dataset = "GSEUNN"

features = ["DNAmAgeAcc", "DNAmAgeHannumAcc", "DNAmPhenoAgeAcc", "DNAmGrimAgeAcc"]

num_cpgs_to_plot = 10

datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

path_save = f"{path}/{platform}/{dataset}/special/005_covid_before_after"
Path(f"{path_save}/cpgs").mkdir(parents=True, exist_ok=True)
Path(f"{path_save}/features").mkdir(parents=True, exist_ok=True)

pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
betas = betas_drop_na(betas)
df = pd.merge(pheno, betas, left_index=True, right_index=True)
df_bef = df.loc[(df['COVID'] == 'before') & (df['Sample_Chronology'] == 1) & (df['ID'] != 'I64_1'), :]
df_aft = df.loc[(df['COVID'] == 'after') & (df['Sample_Chronology'] == 2) & (df['ID'] != 'I64_2'), :]
print(f"before: {df_bef.shape[0]}")
print(f"after: {df_aft.shape[0]}")

result = {'Feature': features}
metrics = ['mw_stat', 'mw_pval']
for m in metrics:
    result[m] = np.zeros(len(features))
for f_id, f in tqdm(enumerate(features), desc='Progress', total=len(features)):
    mw_stat, mw_pval = mannwhitneyu(df_bef[f].values, df_aft[f].values)
    result['mw_stat'][f_id] = mw_stat
    result['mw_pval'][f_id] = mw_pval
result = correct_pvalues(result, ['mw_pval'])
result = pd.DataFrame(result)
result.set_index("Feature", inplace=True)
result.sort_values(['mw_pval'], ascending=[True], inplace=True)
result.to_excel(f"{path_save}/features.xlsx", index=True)

for f_id, (f, row) in enumerate(result.iterrows()):
    fig = go.Figure()
    add_box_trace(fig, df_bef[f].values, 'Before')
    add_box_trace(fig, df_aft[f].values, 'After')
    add_layout(fig, '', "Methylation Level", f"{f}: {row['mw_pval']:0.4e}")
    fig.update_layout({'colorway': ['blue', "red"]})
    save_figure(fig, f"{path_save}/features/{f_id}_{f}_box")
    fig = go.Figure()
    add_violin_trace(fig, df_bef[f].values, 'Before')
    add_violin_trace(fig, df_aft[f].values, 'After')
    add_layout(fig, '', "Methylation Level", f"{f}: {row['mw_pval']:0.4e}")
    fig.update_layout({'colorway': ['blue', 'red']})
    save_figure(fig, f"{path_save}/features/{f_id}_{f}_vio")

cpgs = betas.columns.values
result = {'CpG': cpgs}
result['Gene'] = np.zeros(len(cpgs), dtype=object)
result['Region'] = np.zeros(len(cpgs), dtype=object)
metrics = ['mw_stat', 'mw_pval']
for m in metrics:
    result[m] = np.zeros(len(cpgs))
for cpg_id, cpg in tqdm(enumerate(cpgs), desc='Progress', total=len(cpgs)):
    result['Gene'][cpg_id] = manifest.loc[cpg, 'Gene']
    result['Region'][cpg_id] = manifest.loc[cpg, 'UCSC_RefGene_Group']
    mw_stat, mw_pval = mannwhitneyu(df_bef[cpg].values, df_aft[cpg].values)
    result['mw_stat'][cpg_id] = mw_stat
    result['mw_pval'][cpg_id] = mw_pval
result = correct_pvalues(result, ['mw_pval'])
result = pd.DataFrame(result)
result.set_index("CpG", inplace=True)
result.sort_values(['mw_pval'], ascending=[True], inplace=True)
result.to_excel(f"{path_save}/cpgs.xlsx", index=True)

cols = ['mw_pval', 'mw_pval_fdr_bh', 'mw_pval_bonferroni']
for c in cols:
    tmp_df = result.loc[(result[c] < 0.05), :]
    tmp_genes = get_genes_list(tmp_df, 'Gene', ['non-genic'])
    np.savetxt(f"{path_save}/genes_{c}.txt", tmp_genes, fmt="%s")

result = result.head(num_cpgs_to_plot)
for cpg_id, (cpg, row) in enumerate(result.iterrows()):
    fig = go.Figure()
    add_box_trace(fig, df_bef[cpg].values, 'Before')
    add_box_trace(fig, df_aft[cpg].values, 'After')
    add_layout(fig, '', "Methylation Level", f"{cpg} ({manifest.loc[cpg, 'Gene']}): {row['mw_pval']:0.4e}")
    fig.update_layout({'colorway': ['blue', "red"]})
    save_figure(fig, f"{path_save}/cpgs/{cpg_id}_{cpg}_box")
    fig = go.Figure()
    add_violin_trace(fig, df_bef[cpg].values, 'Before')
    add_violin_trace(fig, df_aft[cpg].values, 'After')
    add_layout(fig, '', "Methylation Level", f"{cpg} ({manifest.loc[cpg, 'Gene']}): {row['mw_pval']:0.4e}")
    fig.update_layout({'colorway': ['blue', 'red']})
    save_figure(fig, f"{path_save}/cpgs/{cpg_id}_{cpg}_vio")



