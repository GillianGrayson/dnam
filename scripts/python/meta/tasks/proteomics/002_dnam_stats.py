import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from scripts.python.routines.manifest import get_manifest
from scripts.python.EWAS.routines.correction import correct_pvalues
from tqdm import tqdm


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

folder_name = f"proteomics"
path_save = f"{path}/meta/tasks/{folder_name}"

tissues = ['Brain', 'Liver', 'Blood']

platform = 'GPL13534'
manifest = get_manifest(platform)

for tissue in tissues:
    tmp_path = f"{path_save}/{tissue}"

    betas = pd.read_pickle(f"{tmp_path}/betas.pkl")
    pheno = pd.read_pickle(f"{tmp_path}/pheno.pkl")
    df = pd.merge(pheno, betas, left_index=True, right_index=True)
    df_f = df.loc[df['Sex'].isin(['F']), :]
    df_m = df.loc[df['Sex'].isin(['M']), :]

    cpgs = betas.columns.values

    result = {'CpG': cpgs}
    result['Gene'] = np.zeros(len(cpgs), dtype=object)
    metrics = ['pearson_r', 'pearson_pval', 'spearman_r', 'spearman_pval', 'mannwhitney_stat', 'mannwhitney_pval']
    for m in metrics:
        result[m] = np.zeros(len(cpgs))

    for cpg_id, cpg in tqdm(enumerate(cpgs), desc=f'{tissue}', total=len(cpgs)):
        result['Gene'][cpg_id] = manifest.loc[cpg, 'Gene']
        pearson_r, pearson_pval = pearsonr(df[cpg].values, df['Age'].values)
        result['pearson_r'][cpg_id] = pearson_r
        result['pearson_pval'][cpg_id] = pearson_pval
        spearman_r, spearman_pval = spearmanr(df[cpg].values, df['Age'].values)
        result['spearman_r'][cpg_id] = spearman_r
        result['spearman_pval'][cpg_id] = spearman_pval
        mannwhitney_stat, mannwhitney_pval = mannwhitneyu(df_f[cpg].values, df_m[cpg].values)
        result['mannwhitney_stat'][cpg_id] = mannwhitney_stat
        result['mannwhitney_pval'][cpg_id] = mannwhitney_pval

    result = correct_pvalues(result, ['pearson_pval', 'spearman_pval', 'mannwhitney_pval'])
    result = pd.DataFrame(result)
    result.set_index("CpG", inplace=True)
    result.sort_values([f"mannwhitney_pval"], ascending=[True], inplace=True)
    result.to_excel(f"{tmp_path}/stats.xlsx", index=True)
    result.to_pickle(f"{tmp_path}/stats.pkl")