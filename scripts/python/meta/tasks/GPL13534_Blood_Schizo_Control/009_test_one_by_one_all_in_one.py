import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
import pathlib
from scripts.python.meta.tasks.GPL13534_Blood_ICD10_V.routines import KW_Control
from tqdm import tqdm


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
manifest = get_manifest('GPL13534')
dataset_statuses = {
    'GSE87571': ['Control'],
}

task_name = f"GPL13534_Blood_Schizo_Control"
path_wd = f"{path}/meta/tasks/{task_name}"

pheno_all = pd.read_pickle(f"{path_wd}/origin/pheno_all.pkl")

for d_id, dataset in enumerate(dataset_statuses):
    print(dataset)

    mvals = pd.read_csv(f"{path_wd}/one_by_one/mvals_{dataset}_regRCPqn.txt", delimiter="\t", index_col='ID_REF')
    mvals = mvals.T
    mvals.index.name = "subject_id"
    mvals = mvals.astype('float32')
    pheno = pd.read_pickle(f"{path_wd}/origin/pheno_{dataset}.pkl")
    print(f"Number of total subjects: {pheno.shape[0]}")
    print(f"Number of total CpGs: {mvals.shape[1]}")
    cpgs = mvals.columns.values
    feats = pheno.columns.values
    df = pd.merge(pheno, mvals, left_index=True, right_index=True)
    pheno_test = df.loc[:, feats]
    mvals_test = df.loc[:, cpgs]
    pheno_test.to_pickle(f"{path_wd}/one_by_one/pheno_{dataset}.pkl")
    pheno_test.to_excel(f"{path_wd}/one_by_one/pheno_{dataset}.xlsx", index=True)
    mvals_test.to_pickle(f"{path_wd}/one_by_one/mvals_{dataset}.pkl")

    mvals = pd.read_csv(f"{path_wd}/all_in_one/mvals_{dataset}_regRCPqn.txt", delimiter="\t", index_col='ID_REF')
    mvals = mvals.T
    mvals.index.name = "subject_id"
    mvals = mvals.astype('float32')
    pheno = pd.read_pickle(f"{path_wd}/origin/pheno_{dataset}.pkl")
    print(f"Number of total subjects: {pheno.shape[0]}")
    print(f"Number of total CpGs: {mvals.shape[1]}")
    cpgs = mvals.columns.values
    feats = pheno.columns.values
    df = pd.merge(pheno, mvals, left_index=True, right_index=True)
    pheno_test = df.loc[:, feats]
    mvals_test = df.loc[:, cpgs]
    pheno_test.to_pickle(f"{path_wd}/all_in_one/pheno_{dataset}.pkl")
    pheno_test.to_excel(f"{path_wd}/all_in_one/pheno_{dataset}.xlsx", index=True)
    mvals_test.to_pickle(f"{path_wd}/all_in_one/mvals_{dataset}.pkl")

