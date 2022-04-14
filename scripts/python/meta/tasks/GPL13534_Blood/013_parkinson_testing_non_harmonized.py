import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
import pathlib


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
manifest = get_manifest('GPL13534')

disease = "Parkinson"
dataset_statuses = {
    'GSE145361': ['Control', 'Parkinson'],
    'GSE111629': ['Control', 'Parkinson'],
    'GSE72774': ['Control', 'Parkinson'],
}
datasets_trn_val = ['GSE145361', 'GSE111629']
datasets_tst = ['GSE72774', 'GSE87571']

task_name = f"to_delete/{disease}"
path_wd = f"{path}/meta/tasks/{task_name}"
pathlib.Path(f"{path_wd}/non_harmonized").mkdir(parents=True, exist_ok=True)

for dataset in dataset_statuses:
    print(dataset)
    pheno = pd.read_pickle(f"{path_wd}/origin/pheno_{dataset}.pkl")
    mvals = pd.read_pickle(f"{path_wd}/origin/mvalsT_{dataset}.pkl")
    mvals = mvals.T
    mvals.index.name = "subject_id"
    mvals = mvals.astype('float32')
    pheno_cols = pheno.columns.values
    mvals_cols = mvals.columns.values
    df = pd.merge(pheno, mvals, left_index=True, right_index=True)
    pheno = df.loc[:, pheno_cols]
    mvals = df.loc[:, mvals_cols]
    df.to_pickle(f"{path_wd}/non_harmonized/data_{dataset}.pkl")
    pheno.to_excel(f"{path_wd}/non_harmonized/pheno_{dataset}.xlsx", index=True)

pheno_combo = pd.DataFrame()
pheno_combo.index.name = 'subject_id'
mvals_combo = pd.DataFrame()
for d_id, dataset in enumerate(datasets_trn_val):
    print(dataset)
    pheno = pd.read_pickle(f"{path_wd}/origin/pheno_{dataset}.pkl")
    mvals = pd.read_pickle(f"{path_wd}/origin/mvalsT_{dataset}.pkl")
    mvals = mvals.T
    mvals.index.name = "subject_id"
    mvals = mvals.astype('float32')
    pheno_cols = pheno.columns.values
    mvals_cols = mvals.columns.values
    df = pd.merge(pheno, mvals, left_index=True, right_index=True)

    print(f"Number of subjects in {dataset}: {mvals.shape[0]}")
    print(f"Number of CpGs in {dataset}: {mvals.shape[1]}")

    pheno = df.loc[:, pheno_cols]
    pheno_combo = pd.concat([pheno_combo, pheno], verify_integrity=True)

    mvals = df.loc[:, mvals_cols].T
    if d_id == 0:
        mvals_combo = mvals
    else:
        mvals_combo = mvals_combo.merge(mvals, how='inner', left_index=True, right_index=True)

mvals_combo = mvals_combo.T
mvals_combo.index.name = "subject_id"
mvals_combo = mvals_combo.astype('float32')
print(f"Number of combo subjects: {mvals_combo.shape[0]}")
print(f"Number of combo CpGs: {mvals_combo.shape[1]}")
pheno_combo, mvals_combo = get_pheno_betas_with_common_subjects(pheno_combo, mvals_combo)
cpgs = mvals_combo.columns.values
feats = pheno_combo.columns.values
df_combo = pd.merge(pheno_combo, mvals_combo, left_index=True, right_index=True)
pheno_combo = df_combo.loc[:, feats]
df_combo.to_pickle(f"{path_wd}/non_harmonized/data_GSE145361_GSE111629.pkl")
pheno_combo.to_excel(f"{path_wd}/non_harmonized/pheno_GSE145361_GSE111629.xlsx", index=True)



