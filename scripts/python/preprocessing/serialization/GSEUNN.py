import pandas as pd
from scripts.python.preprocessing.serialization.routines.filter import get_forbidden_cpgs, manifest_filter
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.manifest import get_manifest


dataset = "GSEUNN"
platform = "GPL21145"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
forbidden_types = ["NoCG", "SNP", "MultiHit", "XY"]

manifest = get_manifest(platform)

fn = f"{path}/{platform}/{dataset}/part(passed_408).xlsx"
df = pd.read_excel(fn)
df['tmp'] = df['Sample_Name']
pheno = df.set_index('tmp')
pheno.index.name = "index"

fn = f"{path}/{platform}/{dataset}/beta_funnorm_filtered.txt"
df = pd.read_csv(fn, delimiter="\t", index_col='CpG')
df.index.name = 'CpG'
betas = df.T
betas.index.name = "index"
betas = manifest_filter(betas, manifest)
forbidden_cpgs = get_forbidden_cpgs(f"{path}/{platform}/manifest/forbidden_cpgs", forbidden_types)
betas = betas.loc[:, ~betas.columns.isin(forbidden_cpgs)]

pheno, betas = get_pheno_betas_with_common_subjects(pheno, betas)
if list(pheno.index.values) == list(betas.index.values):
    print("Change index")
    pheno.set_index('index', inplace=True, verify_integrity=False)
    pheno.index.name = "index"
    betas.set_index(pheno.index, inplace=True, verify_integrity=False)
    betas.index.name = "index"
save_pheno_betas_to_pkl(pheno, betas, f"{path}/{platform}/{dataset}")