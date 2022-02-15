import pandas as pd
from scripts.python.preprocessing.serialization.routines.filter import get_forbidden_cpgs, manifest_filter, betas_pvals_filter
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.manifest import get_manifest


dataset = "GSE116378"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
forbidden_types = ["NoCG", "SNP", "MultiHit", "XY"]

datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

fn = f"{path}/{platform}/{dataset}/pheno.xlsx"
df = pd.read_excel(fn)
df[['subject_id', 'title_part_2']] = df['title'].str.split(': ', 1, expand=True)
df['title_part_1'] = df['subject_id']
pheno = df.set_index('subject_id')
pheno.index.name = "subject_id"

fn = f"{path}/{platform}/{dataset}/raw/GSE116378_DUTCHSCZ_Matrix_processed_betas.csv"
df = pd.read_csv(fn, delimiter=",")
df.rename(columns={df.columns[0]: 'CpG'}, inplace=True)
df.set_index('CpG', inplace=True)
betas = df.iloc[:, 0::2]
pvals = df.iloc[:, 1::2]
betas = betas.T
pvals = pvals.T
pvals.index = betas.index.values.tolist()
betas.index.name = "subject_id"
pvals.index.name = "subject_id"
betas = betas_pvals_filter(betas, pvals, 0.01, 0.1)
betas = manifest_filter(betas, manifest)
forbidden_cpgs = get_forbidden_cpgs(f"{path}/{platform}/manifest/forbidden_cpgs", forbidden_types)
betas = betas.loc[:, ~betas.columns.isin(forbidden_cpgs)]

pheno, betas = get_pheno_betas_with_common_subjects(pheno, betas)
if list(pheno.index.values) == list(betas.index.values):
    print("Change index")
    pheno.set_index('geo_accession', inplace=True)
    pheno.index.name = "subject_id"
    betas.set_index(pheno.index, inplace=True)
    betas.index.name = "subject_id"
save_pheno_betas_to_pkl(pheno, betas, f"{path}/{platform}/{dataset}")
