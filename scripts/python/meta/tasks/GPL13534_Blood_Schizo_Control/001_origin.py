import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict_default
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.betas import betas_drop_na
from scripts.python.routines.mvals import logit2
import json
import pathlib


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
manifest = get_manifest('GPL13534')
dataset_statuses = {
    'GSE84727': ['Control', 'Schizophrenia'],
    'GSE152027': ['Control', 'Schizophrenia'],
    'GSE80417': ['Control', 'Schizophrenia'],
    'GSE116379': ['Control', 'Schizophrenia'],
    'GSE41169': ['Control', 'Schizophrenia'],
    'GSE116378': ['Control', 'Schizophrenia'],
}
datasets_train_val = ['GSE152027', 'GSE84727', 'GSE80417']
datasets_test = ['GSE116379', 'GSE41169', 'GSE116378']

task_name = f"GPL13534_Blood_Schizo_Control"
path_wd = f"{path}/meta/tasks/{task_name}"
pathlib.Path(f"{path_wd}/origin").mkdir(parents=True, exist_ok=True)

pheno_all = pd.DataFrame()
pheno_all.index.name = 'subject_id'
for d_id, dataset in enumerate(dataset_statuses):
    print(dataset)
    platform = datasets_info.loc[dataset, 'platform']
    status_col = get_column_name(dataset, 'Status')
    status_dict = get_status_dict_default(dataset)
    status_passed_fields = get_passed_fields(status_dict, dataset_statuses[dataset])
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

    print(f"Number of subjects in {dataset}: {pheno.shape[0]}")
    print(f"Number of CpGs in {dataset}: {betas.shape[1]}")

    pheno_all = pheno_all.append(pheno, verify_integrity=True)

    cpgs = betas.columns.values
    betas = df[cpgs].T
    if d_id == 0:
        betas_all = betas
    else:
        betas_all = betas_all.merge(betas, how='inner', left_index=True, right_index=True)

betas_all = betas_all.T
betas_all.index.name = "subject_id"
betas_all = betas_all.astype('float32')
print(f"Number of total subjects: {pheno_all.shape[0]}")
print(f"Number of total CpGs: {betas_all.shape[1]}")

pheno_all, betas_all = get_pheno_betas_with_common_subjects(pheno_all, betas_all)
pheno_all.to_pickle(f"{path_wd}/origin/pheno_all.pkl")
pheno_all.to_excel(f"{path_wd}/origin/pheno_all.xlsx", index=True)
alpha_all = betas_all.values[betas_all.values > 0].min() * 0.001
mvals_all = logit2(betas_all, alpha_all)
mvals_all = mvals_all.T
mvals_all.index.name = "ID_REF"
mvals_all.to_pickle(f"{path_wd}/origin/mvalsT_all.pkl")

cpgs = betas_all.columns.values
feats = pheno_all.columns.values

df_all = pd.merge(pheno_all, betas_all, left_index=True, right_index=True)
df_train_val = df_all.loc[df_all['Dataset'].isin(datasets_train_val), :]
pheno_train_val = df_train_val.loc[:, feats]
betas_train_val = df_train_val.loc[:, cpgs]
alpha_train_val = betas_train_val.values[betas_train_val.values > 0].min() * 0.001
mvals_train_val = logit2(betas_train_val, alpha_train_val)

pheno_train_val.to_pickle(f"{path_wd}/origin/pheno_train_val.pkl")
pheno_train_val.to_excel(f"{path_wd}/origin/pheno_train_val.xlsx", index=True)
mvals_train_val = mvals_train_val.T
mvals_train_val.index.name = "ID_REF"
mvals_train_val.to_pickle(f"{path_wd}/origin/mvalsT_train_val.pkl")

for dataset in datasets_train_val + datasets_test:
    print(dataset)
    pheno_i = df_all.loc[df_all['Dataset'] == dataset, feats]
    betas_i = df_all.loc[df_all['Dataset'] == dataset, cpgs]
    mvals_i = logit2(betas_i, alpha_train_val)

    pheno_i.to_pickle(f"{path_wd}/origin/pheno_{dataset}.pkl")
    pheno_i.to_excel(f"{path_wd}/origin/pheno_{dataset}.xlsx", index=True)
    mvals_i = mvals_i.T
    mvals_i.index.name = "ID_REF"
    mvals_i.to_pickle(f"{path_wd}/origin/mvalsT_{dataset}.pkl")
