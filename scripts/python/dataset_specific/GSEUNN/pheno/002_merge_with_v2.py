import pandas as pd
from scripts.python.routines.manifest import get_manifest


dataset = "GSEUNN"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

pheno_xtd = pd.read_excel(f"{path}/{platform}/{dataset}/pheno_xtd.xlsx", index_col="subject_id")

pheno_v2 = pd.read_excel(f"E:/YandexDisk/Work/pydnameth/unn_epic/all_data/table_part(v2).xlsx")
pheno_v2['subject_id'] = 'X' + pheno_v2['Sample_Name']
pheno_v2.set_index('subject_id', inplace=True)

cols_to_merge = pheno_v2.columns.difference(pheno_xtd.columns)
pheno = pd.merge(pheno_xtd, pheno_v2[cols_to_merge], left_index=True, right_index=True, how='outer')

pheno.to_excel(f"{path}/{platform}/{dataset}/pheno_xtd.xlsx", index=True)
pheno.to_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
