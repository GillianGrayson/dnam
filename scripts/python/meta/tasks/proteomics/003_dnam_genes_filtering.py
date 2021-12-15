import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from scripts.python.routines.manifest import get_manifest
from scripts.python.EWAS.routines.correction import correct_pvalues
from tqdm import tqdm
import re
import upsetplot as upset
import matplotlib.pyplot as plt


def plot_upset(genes_universe, dict_of_lists, path_save, suffix):
    upset_df = pd.DataFrame(index=list(genes_universe))
    for k, v in dict_of_lists.items():
        upset_df[k] = upset_df.index.isin(v)
    upset_df = upset_df.set_index(list(dict_of_lists.keys()))
    fig = upset.UpSet(upset_df, subset_size='count', show_counts=True, min_degree=1, sort_categories_by=None).plot()
    plt.savefig(f"{path_save}/figs/upset_{suffix}.png", bbox_inches='tight')
    plt.savefig(f"{path_save}/figs/upset_{suffix}.pdf", bbox_inches='tight')


def get_genes_list(df: pd.DataFrame, col: str, emptys):
    genes_raw = df.loc[:, col].values
    genes_all = set()
    for genes_row in genes_raw:
        if genes_row not in emptys:
            genes = set(re.split(r'[.;]+', genes_row))
            genes_all.update(genes)
    return list(genes_all)


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

SS_lists = {}
AA_lists = {}
SSAA_lists = {}

proteomic_path = f"E:/YandexDisk/Work/pydnameth/methylation_and_proteomic"
t1 = pd.read_excel(f"{proteomic_path}/proteomic_data/T1.xlsx", index_col='ID')
t4 = pd.read_excel(f"{proteomic_path}/proteomic_data/T4.xlsx", index_col='ID')
prot = pd.merge(t1, t4, left_index=True, right_index=True)
SS_prot = prot.loc[prot['q.Sex'] < 0.05, :]
SS_prot_genes = get_genes_list(SS_prot, 'EntrezGeneSymbol', [np.nan])
SS_lists['Proteomic'] = SS_prot_genes
print(f"Proteomic SS genes: {len(SS_prot_genes)}")
AA_prot = prot.loc[prot['q.Age'] < 0.05, :]
AA_prot_genes = get_genes_list(AA_prot, 'EntrezGeneSymbol', [np.nan])
AA_lists['Proteomic'] = AA_prot_genes
print(f"Proteomic AA genes: {len(AA_prot_genes)}")
SSAA_prot = prot.loc[(prot['q.Sex'] < 0.05) & (prot['q.Age'] < 0.05), :]
SSAA_prot_genes = get_genes_list(SSAA_prot, 'EntrezGeneSymbol', [np.nan])
SSAA_lists['Proteomic'] = SSAA_prot_genes
print(f"Proteomic SS genes: {len(SSAA_prot_genes)}")

genes_universe = set(get_genes_list(prot, 'EntrezGeneSymbol', [np.nan]))

for tissue in tissues:
    tmp_path = f"{path_save}/{tissue}"

    metrics = ['pearson_r', 'pearson_pval', 'spearman_r', 'spearman_pval', 'mannwhitney_stat', 'mannwhitney_pval']
    corr_types = ['fdr_bh', 'bonferroni']
    stats = pd.read_pickle(f"{tmp_path}/stats.pkl")
    genes_universe.update(set(get_genes_list(stats, 'Gene', ['non-genic'])))

    AA = stats.loc[(stats[f"{age_corr}_pval_{corr_type}"] < thld_age), :]
    AA_genes = get_genes_list(AA, 'Gene', ['non-genic'])
    AA_lists[tissue] = AA_genes
    print(f"{tissue} AA genes: {len(AA_genes)}")

    SS = stats.loc[(stats[f"mannwhitney_pval_{corr_type}"] < thld_sex), :]
    SS_genes = get_genes_list(SS, 'Gene', ['non-genic'])
    SS_lists[tissue] = SS_genes
    print(f"{tissue} SS genes: {len(SS_genes)}")

    SSAA = SS.loc[(SS[f"{age_corr}_pval_{corr_type}"] < thld_age), :]
    SSAA_genes = get_genes_list(SSAA, 'Gene', ['non-genic'])
    SSAA_lists[tissue] = SSAA_genes
    print(f"{tissue} SSAA genes: {len(SSAA_genes)}")

plot_upset(genes_universe, AA_lists, path_save, 'AA')
plot_upset(genes_universe, SS_lists, path_save, 'SS')
plot_upset(genes_universe, SSAA_lists, path_save, 'SSAA')



