import numpy as np
import pandas as pd
from scripts.python.pheno.datasets.features import get_column_name, get_sex_dict
import pathlib


dataset = "GSEUNN"
tissue = 'Blood WB' # "Liver" "Brain FCTX" "Blood WB"
path = f"D:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']

path_save = f"{path}/{platform}/{dataset}/calculator/pc_clock"
pathlib.Path(path_save).mkdir(parents=True, exist_ok=True)

age_col = get_column_name(dataset, 'Age')
sex_col = get_column_name(dataset, 'Sex')
sex_dict = get_sex_dict(dataset)

pheno = pd.read_excel(f"{path}/{platform}/{dataset}/pheno.xlsx", index_col=0)
pheno = pheno[[age_col, sex_col]]
pheno[sex_col] = pheno[sex_col].map({sex_dict["F"]: 1, sex_dict["M"]: 0})
pheno.rename(columns={age_col: 'Age', sex_col: 'Female'}, inplace=True)
pheno["Tissue"] = tissue
pheno.to_pickle(f"{path_save}/pheno.pkl")

cpgs = pd.read_excel(f"{path}/lists/cpgs/PC_clocks.xlsx")["CpG"].values
