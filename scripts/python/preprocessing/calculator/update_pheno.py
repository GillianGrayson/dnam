import pandas as pd


dataset = "GSE152026"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
features = ["DNAmAge", "CD8T", "CD4T", "NK", "Bcell", "Mono", "Gran", "propNeuron", "DNAmAgeHannum", "DNAmPhenoAge", "DNAmGDF15", "DNAmGrimAge", "IEAA", "EEAA", "IEAA.Hannum"]

pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno.pkl")
pheno = pheno.drop(features, axis=1, errors='ignore')

calcs = pd.read_csv(f"{path}/{platform}/{dataset}/calculator/betas.output.csv", delimiter=",", index_col='subject_id')
calcs = calcs[features]

df = pd.merge(pheno, calcs, left_index=True, right_index=True)
df.to_excel(f"{path}/{platform}/{dataset}/pheno_xtd.xlsx")
df.to_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
