import pandas as pd
import pathlib
from scripts.python.meta.tasks.GPL13534_Blood_ICD10_V.routines import KW_Control
from scripts.python.routines.manifest import get_manifest
from tqdm import tqdm


path = f"E:/YandexDisk/Work/pydnameth/datasets"
folder_name = f"GPL13534_Blood_ICD10-V"
path_wd = f"{path}/meta/tasks/{folder_name}/R/one_by_one/classification"
pathlib.Path(f"{path_wd}/KW_Control/mvals/fig").mkdir(parents=True, exist_ok=True)

manifest = get_manifest('GPL13534')

pheno = pd.read_pickle(f"{path_wd}/pheno_regRCPqn.pkl")
mvals = pd.read_pickle(f"{path_wd}/mvals_regRCPqn.pkl")
datasets = pheno['Dataset'].unique()

df_mvals = pd.merge(pheno, mvals, left_index=True, right_index=True)

cpgs_metrics_df = KW_Control(datasets, manifest, df_mvals, mvals.columns.values, f"{path_wd}/KW_Control/mvals", "M value")

for cpg_id, cpg in enumerate(tqdm(mvals.columns.values)):
    cpgs_metrics_df.loc[cpg, "mean"] = df_mvals[cpg].mean()
    cpgs_metrics_df.loc[cpg, "median"] = df_mvals[cpg].median()
cpgs_metrics_df.to_excel(f"{path_wd}/cpgs_metrics.xlsx", index=True)
