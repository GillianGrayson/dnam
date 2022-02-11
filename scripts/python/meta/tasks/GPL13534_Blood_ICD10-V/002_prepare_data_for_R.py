import pandas as pd
import pyreadr
import pathlib


path_global = f"E:/YandexDisk/Work/pydnameth/datasets"
folder_name = f"GPL13534_Blood_ICD10-V"
path = f"{path_global}/meta/tasks/GPL13534_Blood_ICD10-V"
pathlib.Path(f"{path}/R").mkdir(parents=True, exist_ok=True)

betas = pd.read_pickle(f"{path}/betas.pkl")
pheno = pd.read_pickle(f"{path}/pheno.pkl")

pheno.insert(loc=0, column='Sample_Name', value=pheno.index)

betas = betas.T
betas.index.name = "ID_REF"
#betas.insert(loc=0, column='ID_REF', value=betas.index)

# pyreadr.write_rdata(f"{path}/R/betas.RData", betas, df_name="betas")
# betas.to_csv(f"{path}/R/betas.csv", index=True, index_label="ID_REF")
betas.to_pickle(f"{path}/R/betasT.pkl")
pheno.to_pickle(f"{path}/R/pheno.pkl")