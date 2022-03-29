import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
import pathlib
from scripts.python.meta.tasks.GPL13534_Blood.routines import perform_test_for_controls
from tqdm import tqdm


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
manifest = get_manifest('GPL13534')
dataset_statuses = {
    'GSE84727': ['Control', 'Schizophrenia'],
    'GSE80417': ['Control', 'Schizophrenia'],
    'GSE152027': ['Control', 'Schizophrenia'],
    'GSE116379': ['Control', 'Schizophrenia'],
    'GSE41169': ['Control', 'Schizophrenia'],
    'GSE116378': ['Control', 'Schizophrenia'],
    'GSE87571': ['Control'],
}
datasets_trn_val = ['GSE84727', 'GSE80417']
datasets_tst = ['GSE152027', 'GSE116379', 'GSE41169', 'GSE116378', 'GSE87571']

task_name = f"GPL13534_Blood/Schizophrenia"
path_wd = f"{path}/meta/tasks/{task_name}"
pathlib.Path(f"{path_wd}/non_harmonized/cpgs/figs").mkdir(parents=True, exist_ok=True)

# Train/Val data =======================================================================================================
pheno_trn_val = pd.DataFrame()
pheno_trn_val.index.name = 'subject_id'
mvals_trn_val = pd.DataFrame()
for d_id, dataset in enumerate(datasets_trn_val):
    print(dataset)
    pheno_i = pd.read_pickle(f"{path_wd}/origin/pheno_trn_val_{dataset}.pkl")
    pheno_cols = pheno_i.columns.values
    mvals_i = pd.read_pickle(f"{path_wd}/origin/mvalsT_trn_val_{dataset}.pkl")
    mvals_i = mvals_i.T
    mvals_cols = mvals_i.columns.values
    df_i = pd.merge(pheno_i, mvals_i, left_index=True, right_index=True)
    pheno_i = df_i.loc[:, pheno_cols]
    mvals_i = df_i.loc[:, mvals_cols]

    pheno_trn_val = pheno_trn_val.append(pheno_i, verify_integrity=True)

    mvals_i = mvals_i.T
    if d_id == 0:
        mvals_trn_val = mvals_i
    else:
        mvals_trn_val = mvals_trn_val.merge(mvals_i, how='inner', left_index=True, right_index=True)

mvals_trn_val = mvals_trn_val.T
mvals_trn_val.index.name = "subject_id"
mvals_trn_val = mvals_trn_val.astype('float32')
print(f"Number of total subjects: {mvals_trn_val.shape[0]}")
print(f"Number of total CpGs: {mvals_trn_val.shape[1]}")
pheno_trn_val, mvals_trn_val = get_pheno_betas_with_common_subjects(pheno_trn_val, mvals_trn_val)
feats = pheno_trn_val.columns.values
cpgs = mvals_trn_val.columns.values
df_trn_val = pd.merge(pheno_trn_val, mvals_trn_val, left_index=True, right_index=True)
pheno_trn_val = df_trn_val.loc[:, feats]
df_trn_val.to_pickle(f"{path_wd}/non_harmonized/data_trn_val.pkl")
pheno_trn_val.to_excel(f"{path_wd}/non_harmonized/pheno_trn_val.xlsx", index=True)

cpgs_metrics_df = perform_test_for_controls(datasets_trn_val, manifest, df_trn_val, cpgs, f"{path_wd}/non_harmonized/cpgs/figs", "M value")
for cpg_id, cpg in enumerate(tqdm(cpgs)):
    cpgs_metrics_df.loc[cpg, "mean"] = df_trn_val[cpg].mean()
    cpgs_metrics_df.loc[cpg, "median"] = df_trn_val[cpg].median()
cpgs_metrics_df.to_excel(f"{path_wd}/non_harmonized/cpgs/{cpgs_metrics_df.shape[0]}.xlsx", index=True)

# Test data ============================================================================================================
for d_id, dataset in enumerate(datasets_tst):
    print(dataset)
    pheno = pd.read_pickle(f"{path_wd}/origin/pheno_tst_{dataset}.pkl")
    feats = pheno.columns.values
    mvals = pd.read_pickle(f"{path_wd}/origin/mvalsT_tst_{dataset}.pkl")
    mvals = mvals.T
    mvals = mvals.astype('float32')
    mvals.index.name = "subject_id"
    cpgs = mvals.columns.values
    print(f"Number of total subjects: {pheno.shape[0]}")
    print(f"Number of total CpGs: {mvals.shape[1]}")
    df = pd.merge(pheno, mvals, left_index=True, right_index=True)
    df.to_pickle(f"{path_wd}/non_harmonized/data_tst_{dataset}.pkl")
    pheno_test = df.loc[:, feats]
    pheno_test.to_excel(f"{path_wd}/non_harmonized/pheno_tst_{dataset}.xlsx", index=True)
