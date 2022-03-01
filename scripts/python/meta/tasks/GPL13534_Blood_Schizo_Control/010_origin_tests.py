import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
import pathlib
from scripts.python.meta.tasks.GPL13534_Blood_ICD10_V.routines import KW_Control
from tqdm import tqdm


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
manifest = get_manifest('GPL13534')
dataset_statuses = {
    'GSE152027': ['Control', 'Schizophrenia'],
    'GSE84727': ['Control', 'Schizophrenia'],
    'GSE80417': ['Control', 'Schizophrenia'],
    'GSE116379': ['Control', 'Schizophrenia'],
    'GSE41169': ['Control', 'Schizophrenia'],
    'GSE116378': ['Control', 'Schizophrenia'],
}
datasets_train_val = ['GSE152027', 'GSE84727', 'GSE80417']
datasets_test = ['GSE116379', 'GSE41169', 'GSE116378']

task_name = f"GPL13534_Blood_Schizo_Control"
path_wd = f"{path}/meta/tasks/{task_name}"
pathlib.Path(f"{path_wd}/origin/KW/fig").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_wd}/origin/cpgs").mkdir(parents=True, exist_ok=True)

pheno_all = pd.read_pickle(f"{path_wd}/origin/pheno_all.pkl")
mvals_all = pd.read_pickle(f"{path_wd}/origin/mvalsT_all.pkl")
mvals_all = mvals_all.T
cpgs = mvals_all.columns.values
feats = pheno_all.columns.values
df_all = pd.merge(pheno_all, mvals_all, left_index=True, right_index=True)

for d_id, dataset in enumerate(dataset_statuses):
    print(dataset)
    df_i = df_all.loc[df_all['Dataset'] == dataset, :]
    print(df_i.shape)

cpgs_metrics_df = KW_Control(datasets_train_val, manifest, df_all, cpgs, f"{path_wd}/origin/KW", "M value")
for cpg_id, cpg in enumerate(tqdm(cpgs)):
    cpgs_metrics_df.loc[cpg, "mean"] = df_all[cpg].mean()
    cpgs_metrics_df.loc[cpg, "median"] = df_all[cpg].median()
cpgs_metrics_df.to_excel(f"{path_wd}/origin/cpgs/{cpgs_metrics_df.shape[0]}.xlsx", index=True)
