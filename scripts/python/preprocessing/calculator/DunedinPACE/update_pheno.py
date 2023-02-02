import pandas as pd

dataset = "GSEUNN"
path = f"D:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']

pheno = pd.read_excel(f"{path}/{platform}/{dataset}/pheno.xlsx", index_col=0)

calcs = pd.read_excel(f"{path}/{platform}/{dataset}/calculator/DunedinPACE/result.xlsx", index_col=0)

pheno.loc[calcs.index.values, calcs.columns.values] = calcs.loc[calcs.index.values, calcs.columns.values]

pheno.index.name = 'index'
if not pheno.index.is_unique:
    raise ValueError("Non-unique index")
pheno.to_excel(f"{path}/{platform}/{dataset}/pheno_1.xlsx")
