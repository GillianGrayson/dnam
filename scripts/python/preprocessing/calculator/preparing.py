import numpy as np
import pandas as pd
import os
from scripts.python.pheno.datasets.features import get_column_name, get_sex_dict

dataset = "GSE55763"
tissue = 'Blood WB' # "Liver" "Brain FCTX" "Blood WB"
path = f"D:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']

age_col = get_column_name(dataset, 'Age')
sex_col = get_column_name(dataset, 'Sex')
sex_dict = get_sex_dict(dataset)

save_path = f"{path}/{platform}/{dataset}/calculator"
if not os.path.exists(save_path):
    os.makedirs(save_path)

pheno = pd.read_excel(f"{path}/{platform}/{dataset}/pheno.xlsx", index_col="subject_id")
pheno = pheno[[age_col, sex_col]]
pheno[sex_col] = pheno[sex_col].map({sex_dict["F"]: 1, sex_dict["M"]: 0})
pheno.rename(columns={age_col: 'Age', sex_col: 'Female'}, inplace=True)
pheno["Tissue"] = tissue
pheno.to_csv(f"{save_path}/pheno.csv", na_rep="NA")

with open(f"{path}/lists/cpgs/cpgs_horvath_calculator.txt") as f:
    cpgs_h = f.read().splitlines()
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
betas.set_index(pheno.index, inplace=True)
cpgs_na = list(set(cpgs_h) - set(betas.columns.values))
betas = betas[betas.columns.intersection(cpgs_h)]
betas[cpgs_na] = np.nan
betas = betas.T
betas.index.name = 'ProbeID'
betas.to_csv(f"{save_path}/betas.csv", na_rep="NA")
