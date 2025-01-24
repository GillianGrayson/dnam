import pandas as pd
from scripts.python.preprocessing.serialization.routines.filter import get_forbidden_cpgs, manifest_filter, betas_pvals_filter
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.manifest import get_manifest


dataset = "GSE113725"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
forbidden_types = ["NoCG", "SNP", "MultiHit", "XY"]

datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

fn = f"{path}/{platform}/{dataset}/pheno.xlsx"
df = pd.read_excel(fn)
df['subject_id'] = df['title']
pheno = df.set_index('subject_id')
pheno.index.name = "subject_id"

fn = f"{path}/{platform}/{dataset}/raw/GSE113725_rawBetas.csv"
betas = pd.read_csv(fn, delimiter=",")
betas.rename(columns={betas.columns[0]: 'CpG'}, inplace=True)
betas.set_index('CpG', inplace=True)

fn = f"{path}/{platform}/{dataset}/raw/GSE113725_detectionP.csv"
pvals = pd.read_csv(fn, delimiter=",")
pvals.rename(columns={pvals.columns[0]: 'CpG'}, inplace=True)
pvals.set_index('CpG', inplace=True)

if betas.index.values.tolist() == pvals.index.values.tolist():
    print("CpGs order in betas in pvals is the same")
else:
    raise ValueError("CpGs order in betas in pvals is not the same")

betas = betas.T
pvals = pvals.T

if betas.index.values.tolist() == pvals.index.values.tolist():
    print("Subjects order in betas in pvals is the same")
else:
    raise ValueError("Subjects order in betas in pvals is not the same")

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
