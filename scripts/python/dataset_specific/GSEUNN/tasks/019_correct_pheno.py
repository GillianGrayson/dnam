import pandas as pd
import numpy as np
from scripts.python.pheno.datasets.filter import filter_pheno
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from scripts.python.routines.manifest import get_manifest




dataset = "GSEUNN"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)
pheno_old = pd.read_excel(f"{path}/{platform}/{dataset}/legacy/pheno_xtd.xlsx", index_col='subject_id')
pheno_new = pd.read_excel(f"{path}/{platform}/{dataset}/pheno_xtd.xlsx", index_col='subject_id')
pheno_old_add = pheno_old.loc[~pheno_old.index.isin(pheno_new.index), :]

pheno_all = pheno_new.append(pheno_old_add, verify_integrity=True)

pheno_all.to_pickle(f"{path}/{platform}/{dataset}/pheno_xtd_tmp.pkl")
pheno_all.to_excel(f"{path}/{platform}/{dataset}/pheno_xtd_tmp.xlsx", index=True)
