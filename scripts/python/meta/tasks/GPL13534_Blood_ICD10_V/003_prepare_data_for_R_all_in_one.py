import pandas as pd
import pathlib
from scripts.python.routines.mvals import logit2, expit2


path_global = f"E:/YandexDisk/Work/pydnameth/datasets"
folder_name = f"GPL13534_Blood_ICD10-V"
path = f"{path_global}/meta/tasks/GPL13534_Blood_ICD10-V"
pathlib.Path(f"{path}/R/all_in_one").mkdir(parents=True, exist_ok=True)

betas = pd.read_pickle(f"{path}/betas.pkl")
pheno = pd.read_pickle(f"{path}/pheno.pkl")

pheno.insert(loc=0, column='Sample_Name', value=pheno.index)

betas = betas.T
betas.index.name = "ID_REF"
mvals = logit2(betas)
mvals.index.name = "ID_REF"

betas.to_pickle(f"{path}/R/all_in_one/betasT.pkl")
mvals.to_pickle(f"{path}/R/all_in_one/mvalsT.pkl")
pheno.to_pickle(f"{path}/R/all_in_one/pheno.pkl")
