import pandas as pd
from scripts.python.routines.manifest import get_manifest
import pathlib
from scripts.python.routines.mvals import logit2, expit2
from scripts.python.meta.tasks.GPL13534_Blood_ICD10_V.routines import KW_Control


path_global = f"E:/YandexDisk/Work/pydnameth/datasets"
folder_name = f"GPL13534_Blood_ICD10-V"
path = f"{path_global}/meta/tasks/GPL13534_Blood_ICD10-V"

manifest = get_manifest('GPL13534')

pheno = pd.read_pickle(f"{path}/pheno.pkl")
betas = pd.read_pickle(f"{path}/betas.pkl")
mvals = logit2(betas)

datasets = pheno['Dataset'].unique()

pathlib.Path(f"{path}/KW_Control_for_initilal_data/betas/fig").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path}/KW_Control_for_initilal_data/mvals/fig").mkdir(parents=True, exist_ok=True)

df_betas = pd.merge(pheno, betas, left_index=True, right_index=True)
df_mvals = pd.merge(pheno, mvals, left_index=True, right_index=True)

KW_Control(datasets, manifest, df_betas, betas.columns.values, f"{path}/KW_Control_for_initilal_data/betas", "Beta value")
KW_Control(datasets, manifest, df_mvals, mvals.columns.values, f"{path}/KW_Control_for_initilal_data/mvals", "M value")
