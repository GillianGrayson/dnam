import copy

import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
import os
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_statuses_datasets_dict
from sklearn.feature_selection import VarianceThreshold
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.betas import betas_drop_na
import hashlib
import pickle
import json
import pathlib


dataset = "GSE52588"

path = f"E:/YandexDisk/Work/pydnameth/datasets"
path_background = "E:/YandexDisk/Work/pydnameth/datasets/meta/121da597d6d3fe7b3b1b22a0ddc26e61"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

path_save = f"{path}/{platform}/{dataset}/special/001_generate_dataset"
pathlib.Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

status_col = get_column_name(dataset, 'Status').replace(' ','_')
status_dict = get_status_dict(dataset)
status_passed_fields = status_dict['Control'] + status_dict['Case']
continuous_vars = {}
categorical_vars = {status_col: [x.column for x in status_passed_fields]}
pheno = pd.read_excel(f"{path}/{platform}/{dataset}/pheno.xlsx", index_col="subject_id")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
betas = betas_drop_na(betas)

pheno = pheno.loc[:, [status_col]]
status_dict_inverse = dict((x.column, x.label) for x in status_passed_fields)
pheno[status_col].replace(status_dict_inverse, inplace=True)
pheno.rename(columns={status_col: 'Status'}, inplace=True)

cpgs_target_fn = f"{path_background}/cpgs/24829/tabnetpl/average/all/340.xlsx"
cpgs_df = pd.read_excel(cpgs_target_fn)
cpgs_target = cpgs_df.loc[:, 'CpG'].values

missed_cpgs = list(set(cpgs_target) - set(betas.columns.values))

if len(missed_cpgs) == 0:
    print(f"There is no missed CpGs")
else:
    betas_background = pd.read_pickle(f"{path_background}/betas.pkl")
    pheno_background = pd.read_pickle(f"{path_background}/pheno.pkl")

    statuses_df = pd.read_excel(f"{path_background}/statuses/4.xlsx")
    statuses_background = {}
    for st_id, st in enumerate(statuses_df.loc[:, "Status"].values):
        statuses_background[st] = st_id
    pheno_background = pheno_background.loc[pheno_background["Status"].isin(statuses_background)]
    pheno_background['Status_Origin'] = pheno_background["Status"]
    pheno_background["Status"].replace(statuses_background, inplace=True)

    betas_background = betas_background.loc[pheno_background.index.values, missed_cpgs]

    for cpg in missed_cpgs:
        betas[cpg] =  betas_background[cpg].median()

betas = betas.loc[:, cpgs_target]

pheno.to_pickle(f"{path_save}/pheno.pkl")
pheno.to_excel(f"{path_save}/pheno.xlsx", index=True)
betas.to_pickle(f"{path_save}/betas.pkl")
