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

# disease = "Schizophrenia"
# dataset_statuses = {
#     'GSE84727': ['Control', 'Schizophrenia'],
#     'GSE80417': ['Control', 'Schizophrenia'],
#     'GSE152027': ['Control', 'Schizophrenia'],
#     'GSE116379': ['Control', 'Schizophrenia'],
#     'GSE41169': ['Control', 'Schizophrenia'],
#     'GSE116378': ['Control', 'Schizophrenia'],
#     'GSE87571': ['Control'],
# }
# datasets_trn_val = ['GSE84727', 'GSE80417']
# datasets_tst = ['GSE152027', 'GSE116379', 'GSE41169', 'GSE116378', 'GSE87571']

disease = "Parkinson"
dataset_statuses = {
    'GSE145361': ['Control', 'Parkinson'],
    'GSE111629': ['Control', 'Parkinson'],
    'GSE72774': ['Control', 'Parkinson'],
    'GSE87571': ['Control'],
}
datasets_trn_val = ['GSE145361', 'GSE111629']
datasets_tst = ['GSE72774', 'GSE87571']

alpha = 1e-9

task_name = f"GPL13534_Blood/{disease}"
path_wd = f"{path}/meta/tasks/{task_name}"
pathlib.Path(f"{path_wd}/origin/cpgs/figs").mkdir(parents=True, exist_ok=True)

# Train/Val data =======================================================================================================
pheno_trn_val = pd.DataFrame()
pheno_trn_val.index.name = 'subject_id'
betas_trn_val = pd.DataFrame()
for d_id, dataset in enumerate(datasets_trn_val):
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

    pheno_trn_val = pheno_trn_val.append(pheno, verify_integrity=True)

    cpgs_i = betas.columns.values
    betas = df[cpgs_i].T
    if d_id == 0:
        betas_trn_val = betas
    else:
        betas_trn_val = betas_trn_val.merge(betas, how='inner', left_index=True, right_index=True)

betas_trn_val = betas_trn_val.T
betas_trn_val.index.name = "subject_id"
betas_trn_val = betas_trn_val.astype('float32')
print(f"Number of train/val subjects: {betas_trn_val.shape[0]}")
print(f"Number of train/val CpGs: {betas_trn_val.shape[1]}")
pheno_trn_val, betas_trn_val = get_pheno_betas_with_common_subjects(pheno_trn_val, betas_trn_val)
cpgs = betas_trn_val.columns.values
feats = pheno_trn_val.columns.values
df_trn_val = pd.merge(pheno_trn_val, betas_trn_val, left_index=True, right_index=True)
for dataset in datasets_trn_val:
    print(dataset)
    pheno_i = df_trn_val.loc[df_trn_val['Dataset'] == dataset, feats]
    betas_i = df_trn_val.loc[df_trn_val['Dataset'] == dataset, cpgs]
    mvals_i = logit2(betas_i, alpha)
    pheno_i.to_pickle(f"{path_wd}/origin/pheno_trn_val_{dataset}.pkl")
    pheno_i.to_excel(f"{path_wd}/origin/pheno_trn_val_{dataset}.xlsx", index=True)
    mvals_i = mvals_i.T
    mvals_i.index.name = "ID_REF"
    mvals_i.to_pickle(f"{path_wd}/origin/mvalsT_trn_val_{dataset}.pkl")

cpgs_metrics_df = perform_test_for_controls(datasets_trn_val, manifest, df_trn_val, cpgs, f"{path_wd}/origin/cpgs/figs", "Beta value")
for cpg_id, cpg in enumerate(tqdm(cpgs)):
    cpgs_metrics_df.loc[cpg, "mean"] = df_trn_val[cpg].mean()
    cpgs_metrics_df.loc[cpg, "median"] = df_trn_val[cpg].median()
cpgs_metrics_df.to_excel(f"{path_wd}/origin/cpgs/{cpgs_metrics_df.shape[0]}.xlsx", index=True)

# Test data ============================================================================================================
for d_id, dataset in enumerate(datasets_tst):
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
    betas = betas.astype('float32')
    df = pd.merge(pheno, betas, left_index=True, right_index=True)

    pheno = df.loc[:, [status_col]]
    status_dict_inverse = dict((x.column, x.label) for x in status_passed_fields)
    pheno[status_col].replace(status_dict_inverse, inplace=True)
    pheno.rename(columns={status_col: 'Status'}, inplace=True)
    pheno.loc[:, 'Dataset'] = dataset

    betas = df.loc[:, betas.columns.values]
    mvals = logit2(betas, alpha)

    pheno.to_pickle(f"{path_wd}/origin/pheno_tst_{dataset}.pkl")
    pheno.to_excel(f"{path_wd}/origin/pheno_tst_{dataset}.xlsx", index=True)
    mvals = mvals.T
    mvals.index.name = "ID_REF"
    mvals.to_pickle(f"{path_wd}/origin/mvalsT_tst_{dataset}.pkl")
