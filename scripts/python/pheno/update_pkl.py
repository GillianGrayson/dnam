import pandas as pd


dataset = "GSEUNN"
path = f"E:/YandexDisk/Work/pydnameth/datasets"

datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']

pheno = pd.read_excel(f"{path}/{platform}/{dataset}/pheno_xtd.xlsx", index_col="subject_id")
pheno.to_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
