import pandas as pd
from scripts.python.preprocessing.serialization.routines.filter import get_forbidden_cpgs, manifest_filter
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.manifest import get_manifest


dataset = "GSE111629"
platform = "GPL13534"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
forbidden_types = ["NoCG", "SNP", "MultiHit", "XY"]

meth_type = "processed" # "processed"

manifest = get_manifest(platform)

fn = f"{path}/{platform}/{dataset}/pheno.xlsx"
df = pd.read_excel(fn)


if meth_type == "idat":
    pheno = df.set_index('subject_id')
    pheno.index.name = "subject_id"

    fn = f"{path}/{platform}/{dataset}/raw/result/part(all)_config(0.01_0.10_0.10)/beta_funnorm_filtered.txt"
    df = pd.read_csv(fn, delimiter="\t", index_col='CpG')
    betas = df.T
    betas.index.name = "subject_id"
else:
    df['index'] = df['source_name']
    pheno = df.set_index('index')

    fn = f"{path}/{platform}/{dataset}/raw/GSE111629_PEGblood_450kMethylationDataBackgroundNormalized.txt"
    df = pd.read_csv(fn, delimiter="\t")
    df.rename(columns={df.columns[0]: 'CpG'}, inplace=True)
    df.set_index('CpG', inplace=True)
    betas = df.T
    cols_with_na = betas.columns[betas.isna().any()].tolist()
    betas.index.name = "subject_id"

betas = manifest_filter(betas, manifest)
forbidden_cpgs = get_forbidden_cpgs(f"{path}/{platform}/manifest/forbidden_cpgs", forbidden_types)
betas = betas.loc[:, ~betas.columns.isin(forbidden_cpgs)]

pheno, betas = get_pheno_betas_with_common_subjects(pheno, betas)
if list(pheno.index.values) == list(betas.index.values):
    if 'geo_accession' in pheno:
        print("Change index")
        pheno.set_index('geo_accession', inplace=True)
    pheno.index.name = "subject_id"
    betas.set_index(pheno.index, inplace=True)
    betas.index.name = "subject_id"
num_na = betas.isna().sum().sum()
print(f"Number of NaNs in betas: {num_na}")
betas.dropna(axis='columns', how='any', inplace=True)
save_pheno_betas_to_pkl(pheno, betas, f"{path}/{platform}/{dataset}")
