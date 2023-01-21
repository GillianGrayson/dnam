import pandas as pd
import re

dataset = "GSE40279"
path = f"D:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
features = ["DNAmAge", "CD8T", "CD4T", "NK", "Bcell", "Mono", "Gran", "propNeuron", "DNAmAgeHannum", "DNAmPhenoAge", "DNAmGDF15", "DNAmLeptin", "DNAmGrimAge", "IEAA", "EEAA", "IEAA.Hannum", "DNAmAgeSkinBloodClock"]

pheno = pd.read_excel(f"{path}/{platform}/{dataset}/pheno.xlsx", index_col="subject_id")
pheno['Sample_Name'] = pheno.index

pheno = pheno.drop(features, axis=1, errors='ignore')

replace_values = pheno.index.values
chars_to_remove = ['-', '/', ' ', '+', '(', ')']
regular_expression = '[' + re.escape (''. join (chars_to_remove)) + ']'
replace_keys = pheno.index.str.replace(regular_expression, '.', regex=True)
replace_dict = dict(zip(replace_keys, replace_values))

calcs = pd.read_csv(f"{path}/{platform}/{dataset}/calculator/betas.output.csv", delimiter=",", index_col='SampleID')
calcs = calcs[features]
calcs = calcs.rename(replace_dict)

df = pd.merge(pheno, calcs, left_index=True, right_index=True)
df.set_index('Sample_Name', inplace=True, verify_integrity=True)
df.index.name = 'index'
if not df.index.is_unique:
    raise ValueError("Non-unique index")
df.to_excel(f"{path}/{platform}/{dataset}/pheno_1.xlsx")

