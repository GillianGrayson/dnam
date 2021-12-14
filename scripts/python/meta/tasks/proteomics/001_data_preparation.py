import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_default_statuses_ids, get_status_dict, get_default_statuses, get_sex_dict
from sklearn.feature_selection import VarianceThreshold
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.betas import betas_drop_na
import hashlib
import pickle
import plotly.graph_objects as go
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.histogram import add_histogram_trace
from scripts.python.routines.plot.layout import add_layout
import json
from pathlib import Path


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

folder_name = f"proteomics"
path_save = f"{path}/meta/tasks/{folder_name}"
Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

tissue_datasets = {
    'Brain': ['GSE74193'],
    'Liver': ['GSE48325', 'GSE61258', 'GSE61446'],
    'Blood': ['GSE87571']
}

target_features = ['Status', 'Age', 'Sex']

for tissue, datasets in tissue_datasets.items():
    tmp_path = f"{path_save}/{tissue}"
    Path(f"{tmp_path}/figs").mkdir(parents=True, exist_ok=True)

    pheno_all = pd.DataFrame(columns=target_features + ['Dataset'])
    pheno_all.index.name = 'subject_id'
    for d_id, dataset in enumerate(datasets):
        platform = datasets_info.loc[dataset, 'platform']
        manifest = get_manifest(platform)

        statuses = get_default_statuses(dataset)
        status_col = get_column_name(dataset, 'Status').replace(' ', '_')
        statuses_ids = get_default_statuses_ids(dataset)
        status_dict = get_status_dict(dataset)
        status_passed_fields = get_passed_fields(status_dict, statuses)
        controls_status_vals = [status_dict['Control'][x].column for x in statuses_ids['Control']]
        controls_labels = ', '.join([status_dict['Control'][x].label for x in statuses_ids['Control']])

        age_col = get_column_name(dataset, 'Age').replace(' ', '_')

        sex_col = get_column_name(dataset, 'Sex').replace(' ', '_')
        sex_dict = get_sex_dict(dataset)

        continuous_vars = {'Age': age_col}
        categorical_vars = {
            status_col: [x.column for x in status_passed_fields],
            sex_col: [sex_dict[x] for x in sex_dict]
        }

        pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno.pkl")
        pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
        betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
        betas = betas_drop_na(betas)
        df = pd.merge(pheno, betas, left_index=True, right_index=True)
        df = df.loc[df[status_col].isin(controls_status_vals), :]

        pheno = df.loc[:, [status_col, sex_col, age_col]]
        status_dict_inverse = dict((x.column, x.label) for x in status_passed_fields)
        pheno[status_col].replace(status_dict_inverse, inplace=True)
        pheno.rename(columns={status_col: 'Status'}, inplace=True)
        sex_dict_inverse = {v: k for k, v in sex_dict.items()}
        pheno[sex_col].replace(sex_dict_inverse, inplace=True)
        pheno.rename(columns={sex_col: 'Sex'}, inplace=True)
        pheno.rename(columns={age_col: 'Age'}, inplace=True)

        pheno.loc[:, 'Dataset'] = dataset
        pheno_all = pheno_all.append(pheno, verify_integrity=True)

        cpgs = betas.columns.values
        betas = df[cpgs].T

        if d_id == 0:
            betas_all = betas
        else:
            betas_all = betas_all.merge(betas, how='inner', left_index=True, right_index=True)

    print(f"Number of remaining subjects: {pheno_all.shape[0]}")

    betas_all = betas_all.T
    betas_all.index.name = "subject_id"

    pheno_all, betas_all = get_pheno_betas_with_common_subjects(pheno_all, betas_all)
    pheno_all.to_pickle(f"{tmp_path}/pheno.pkl")
    pheno_all.to_excel(f"{tmp_path}/pheno.xlsx", index=True)
    betas_all.to_pickle(f"{tmp_path}/betas.pkl")

    info = {tissue: datasets, "betas.shape": betas_all.shape}
    with open(f"{tmp_path}/info.json", 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)

    pheno_f = pheno_all.loc[pheno_all['Sex'].isin(['F']), :]
    pheno_m = pheno_all.loc[pheno_all['Sex'].isin(['M']), :]
    fig = go.Figure()
    add_histogram_trace(fig, pheno_f['Age'].values, f"Female ({pheno_f.shape[0]})", 5.0)
    add_histogram_trace(fig, pheno_m['Age'].values, f"Male ({pheno_m.shape[0]})", 5.0)
    add_layout(fig, "Age", "Count", f"{tissue}")
    fig.update_layout(colorway=['red', 'blue'], barmode='overlay')
    save_figure(fig, f"{tmp_path}/figs/histogram_Age_Sex")
