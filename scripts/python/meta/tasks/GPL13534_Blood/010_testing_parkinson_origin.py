import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict_default
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.routines.betas import betas_drop_na
from scripts.python.routines.mvals import logit2
from scripts.python.meta.tasks.GPL13534_Blood.routines import perform_test_for_controls
from tqdm import tqdm
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
datasets_tst = ['GSE72774']

alpha = 1e-9

task_name = f"to_delete/{disease}"
path_wd = f"{path}/meta/tasks/{task_name}"
pathlib.Path(f"{path_wd}/origin").mkdir(parents=True, exist_ok=True)

# Before Data

# Train/Val data =======================================================================================================
for dataset in dataset_statuses:
    print(dataset)
    platform = datasets_info.loc[dataset, 'platform']
    status_col = get_column_name(dataset, 'Status')
    status_dict = get_status_dict_default(dataset)
    status_passed_fields = get_passed_fields(status_dict, dataset_statuses[dataset])
    continuous_vars = {}
    categorical_vars = {status_col: [x.column for x in status_passed_fields]}
    pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno.pkl")
    pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)

    status_dict_inverse = dict((x.column, x.label) for x in status_passed_fields)
    pheno[status_col].replace(status_dict_inverse, inplace=True)
    pheno.rename(columns={status_col: 'Status'}, inplace=True)
    pheno.loc[:, 'Dataset'] = dataset

    betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
    betas = betas_drop_na(betas)
    df = pd.merge(pheno, betas, left_index=True, right_index=True)
    df.to_pickle(f"{path_wd}/origin/data_{dataset}.pkl")

    betas = df.loc[:, betas.columns.values]
    pheno = df.loc[:, pheno.columns.values]
    mvals = logit2(betas, alpha)
    pheno.to_pickle(f"{path_wd}/origin/pheno_{dataset}.pkl")
    pheno.to_excel(f"{path_wd}/origin/pheno_{dataset}.xlsx", index=True)
    mvals = mvals.T
    mvals.index.name = "ID_REF"
    mvals.to_pickle(f"{path_wd}/origin/mvalsT_{dataset}.pkl")

