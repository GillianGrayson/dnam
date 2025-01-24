import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_statuses_datasets_dict
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.betas import betas_drop_na
import json
import pathlib


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

statuses = [
    'Schizophrenia',
    'First episode psychosis',
    'Depression',
]
include_controls = True
datasets_control = ['GSE87571']
target_features = ['Status']

statuses_datasets_dict = get_statuses_datasets_dict()
datasets = {}
for s in statuses:
    if s in statuses_datasets_dict:
        for dataset in statuses_datasets_dict[s]:
            if dataset not in datasets:
                if include_controls:
                    datasets[dataset] = ['Control', s]
                else:
                    datasets[dataset] = [s]
            else:
                datasets[dataset].append(s)
for ds in datasets_control:
    datasets[ds] = ['Control']
datasets = dict(sorted(datasets.items()))

info = {"statuses": statuses, "datasets": datasets, 'include_controls': include_controls}

folder_name = f"GPL13534_Blood_ICD10-V"
path_save = f"{path}/meta/tasks/{folder_name}"
pathlib.Path(f"{path_save}/figs/KW").mkdir(parents=True, exist_ok=True)
with open(f"{path_save}/info.json", 'w', encoding='utf-8') as f:
    json.dump(info, f, ensure_ascii=False, indent=4)
manifest = get_manifest('GPL13534')

pheno_all = pd.DataFrame(columns=target_features + ['Dataset'])
pheno_all.index.name = 'subject_id'
for d_id, dataset in enumerate(datasets):
    print(dataset)

    platform = datasets_info.loc[dataset, 'platform']

    curr_statuses = datasets[dataset]

    status_col = get_column_name(dataset, 'Status')
    status_dict = get_status_dict(dataset)
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
save_pheno_betas_to_pkl(pheno_all, betas_all, f"{path_save}")
