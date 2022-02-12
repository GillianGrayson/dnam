import pandas as pd
import pathlib
from scripts.python.meta.tasks.GPL13534_Blood_ICD10_V.routines import KW_Control
from scripts.python.routines.manifest import get_manifest
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects


path_global = f"E:/YandexDisk/Work/pydnameth/datasets"
folder_name = f"GPL13534_Blood_ICD10-V"
path = f"{path_global}/meta/tasks/GPL13534_Blood_ICD10-V"
pathlib.Path(f"{path}/KW_Control_for_R_data_all_in_one/mvals/fig").mkdir(parents=True, exist_ok=True)

manifest = get_manifest('GPL13534')

mvals = pd.read_csv(f"{path}/R/all_in_one/mvals_regRCPqn.txt", delimiter="\t", index_col='ID_REF')
mvals = mvals.astype('float32')
mvals = mvals.T
mvals.index.name = "subject_id"
print(mvals.shape)

pheno = pd.read_pickle(f"{path}/pheno.pkl")

datasets = pheno['Dataset'].unique()

pheno, mvals = get_pheno_betas_with_common_subjects(pheno, mvals)
pheno.to_pickle(f"{path}/R/all_in_one/pheno_regRCPqn.pkl")
pheno.to_excel(f"{path}/R/all_in_one/pheno_regRCPqn.xlsx", index=True)
mvals.to_pickle(f"{path}/R/all_in_one/mvals_regRCPqn.pkl")

df_mvals = pd.merge(pheno, mvals, left_index=True, right_index=True)

KW_Control(datasets, manifest, df_mvals, mvals.columns.values, f"{path}/KW_Control_for_R_data_all_in_one/mvals", "M value")
