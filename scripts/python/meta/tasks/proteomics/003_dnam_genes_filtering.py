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

age_corr = 'spearman' # 'pearson'
corr_type = 'fdr_bh' # 'bonferroni'
thld_age = 0.01
thld_sex = 0.01

def get_genes_list(df: pd.DataFrame):
    genes_raw = df.loc[:, 'Gene'].values
    genes_all = set()
    for genes_row in genes_raw:
        if genes_row != 'non-genic':
            genes = set(genes_row.split(';'))
            genes_all.update(genes)
    return list(genes_all)

for tissue in tissues:
    tmp_path = f"{path_save}/{tissue}"

    metrics = ['pearson_r', 'pearson_pval', 'spearman_r', 'spearman_pval', 'mannwhitney_stat', 'mannwhitney_pval']
    corr_types = ['fdr_bh', 'bonferroni']
    stats = pd.read_pickle(f"{tmp_path}/stats.pkl")

    AA = stats.loc[(stats[f"{age_corr}_pval_{corr_type}"] < thld_age), :]
    AA_genes = get_genes_list(AA)
    print(f"{tissue} AA genes: {len(AA_genes)}")

    SS = stats.loc[(stats[f"mannwhitney_pval_{corr_type}"] < thld_sex), :]
    SS_genes = get_genes_list(SS)
    print(f"{tissue} SS genes: {len(SS_genes)}")

    SSAA = SS.loc[(SS[f"{age_corr}_pval_{corr_type}"] < thld_age), :]
    SSAA_genes = get_genes_list(SSAA)
    print(f"{tissue} SSAA genes: {len(SSAA_genes)}")


