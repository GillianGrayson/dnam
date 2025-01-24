import numpy as np
import pandas as pd
from scripts.python.pheno.datasets.features import get_column_name, get_sex_dict
import pathlib


dataset = "GSE55763"
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
feats_pheno = pheno.columns.values
print(f"pheno: {pheno.shape}")
pheno.dropna(inplace=True)

cpgs = pd.read_excel(f"{path}/lists/cpgs/PC_clocks/cpgs.xlsx")["CpG"].values
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
betas = betas[betas.columns.intersection(cpgs)]
betas = betas.astype('float32')
feats_betas = betas.columns.values
print(f"betas: {betas.shape}")

df = pd.merge(pheno, betas, left_index=True, right_index=True)
pheno = df.loc[:, feats_pheno]
betas = df.loc[:, feats_betas]
print(f"pheno: {pheno.shape}")
print(f"betas: {betas.shape}")

pheno.to_pickle(f"{path_save}/pheno.pkl")
betas = betas.T
betas.index.name = 'ProbeID'
betas.to_pickle(f"{path_save}/betas.pkl")
