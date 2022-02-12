import pandas as pd
import pathlib
from scripts.python.routines.mvals import logit2
import numpy as np


path_global = f"E:/YandexDisk/Work/pydnameth/datasets"
folder_name = f"GPL13534_Blood_ICD10-V"
path = f"{path_global}/meta/tasks/GPL13534_Blood_ICD10-V"
pathlib.Path(f"{path}/R/one_by_one").mkdir(parents=True, exist_ok=True)

betas = pd.read_pickle(f"{path}/betas.pkl")
pheno = pd.read_pickle(f"{path}/pheno.pkl")

max_val = betas.values.max()
min_val = betas.values.min()
pos_min_val = betas.values[betas.values > 0].min()
alpha = pos_min_val * 0.1

datasets = pheno['Dataset'].unique()

df = pd.merge(pheno, betas, left_index=True, right_index=True)

for dataset in datasets:
    df_i = df.loc[df["Dataset"] == dataset, :]
    betas_i = df_i.loc[:, betas.columns]
    pheno_i = df_i.loc[:, pheno.columns]

    pheno_i.insert(loc=0, column='Sample_Name', value=pheno_i.index)

    betas_i = betas_i.T
    betas_i.index.name = "ID_REF"
    print(f"Number of inf values in betas: {np.isinf(betas_i).values.sum()}")
    mvals_i = logit2(betas_i, alpha)
    mvals_i.index.name = "ID_REF"
    print(f"Number of inf values in mvals: {np.isinf(mvals_i).values.sum()}")
    cpgs_with_inf_mvals = mvals_i.index.values[np.isinf(mvals_i).any(1)]
    print(f"Number of CpGs with inf in mvals: {len(cpgs_with_inf_mvals)}")

    mvals_i.to_pickle(f"{path}/R/one_by_one/mvalsT_{dataset}.pkl")
    betas_i.to_pickle(f"{path}/R/one_by_one/betasT_{dataset}.pkl")
    pheno_i.to_pickle(f"{path}/R/one_by_one/pheno_{dataset}.pkl")
