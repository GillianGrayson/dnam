import pandas as pd
import pathlib
from scripts.python.routines.mvals import logit2
import numpy as np
from scripts.python.routines.manifest import get_manifest
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict_default, get_default_statuses_ids
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.betas import betas_drop_na


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

task_name = f"GPL13534_Blood_ICD10-V"
path_save = f"{path}/meta/tasks/{task_name}/test"
pathlib.Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

alpha = 2.4690309146535585e-06

test_datasets = {
    'GSE116379': ['Control', 'Schizophrenia'],
    'GSE113725': ['Control', 'Depression'],
    'GSE41169': ['Control', 'Schizophrenia'],
    'GSE116378': ['Control', 'Schizophrenia'],
}

for d_id, dataset in enumerate(test_datasets):
    print(dataset)
    platform = datasets_info.loc[dataset, 'platform']
    curr_statuses = test_datasets[dataset]

    status_col = get_column_name(dataset, 'Status')
    default_statuses_ids = get_default_statuses_ids(dataset)
    status_dict = get_status_dict_default(dataset)
    status_passed_fields = get_passed_fields(status_dict, curr_statuses)
    continuous_vars = {}
    categorical_vars = {status_col: [x.column for x in status_passed_fields]}
    pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno.pkl")
    pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
    betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
    betas = betas_drop_na(betas)
    df = pd.merge(pheno, betas, left_index=True, right_index=True)

    pheno = df.loc[:, [status_col]]
    status_dict_inverse = dict((x.column, x.label) for x in status_passed_fields)
    pheno[status_col].replace(status_dict_inverse, inplace=True)
    pheno.rename(columns={status_col: 'Status'}, inplace=True)
    pheno.loc[:, 'Dataset'] = dataset

    cpgs = betas.columns.values
    betas = df[cpgs].T

    print(f"Number of subjects in {dataset}: {pheno.shape[0]}")
    print(f"Number of CpGs in {dataset}: {betas.shape[0]}")

    betas.index.name = "ID_REF"
    print(f"Number of inf values in betas: {np.isinf(betas).values.sum()}")
    mvals = logit2(betas, alpha)
    mvals.index.name = "ID_REF"
    print(f"Number of inf values in mvals: {np.isinf(mvals).values.sum()}")
    cpgs_with_inf_mvals = mvals.index.values[np.isinf(mvals).any(1)]
    print(f"Number of CpGs with inf in mvals: {len(cpgs_with_inf_mvals)}")

    mvals.to_pickle(f"{path_save}/mvalsT_{dataset}.pkl")
    pheno.to_pickle(f"{path_save}/pheno_{dataset}.pkl")
    pheno.to_excel(f"{path_save}/pheno_{dataset}.xlsx", index=True)
