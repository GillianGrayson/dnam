import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
import pathlib
from scripts.python.meta.tasks.GPL13534_Blood_ICD10_V.routines import KW_Control
from tqdm import tqdm


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
manifest = get_manifest('GPL13534')
datasets_train_val = ['GSE152027', 'GSE84727', 'GSE80417']
datasets_test = ['GSE116379', 'GSE41169', 'GSE116378']

task_name = f"GPL13534_Blood_Schizo_Control"
path_wd = f"{path}/meta/tasks/{task_name}"
pathlib.Path(f"{path_wd}/all_in_one/KW/fig").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_wd}/all_in_one/cpgs").mkdir(parents=True, exist_ok=True)

pheno_all = pd.read_pickle(f"{path_wd}/origin/pheno_all.pkl")

for d_id, dataset in enumerate(['train_val'] + datasets_test):
    print(dataset)
    mvals = pd.read_csv(f"{path_wd}/all_in_one/mvals_{dataset}_regRCPqn.txt", delimiter="\t", index_col='ID_REF')
    if d_id == 0:
        mvals_all = mvals
    else:
        mvals_all = mvals_all.merge(mvals, how='inner', left_index=True, right_index=True)

mvals_all = mvals_all.T
mvals_all.index.name = "subject_id"
mvals_all = mvals_all.astype('float32')
print(f"Number of total subjects: {pheno_all.shape[0]}")
print(f"Number of total CpGs: {mvals_all.shape[1]}")

pheno_all, mvals_all = get_pheno_betas_with_common_subjects(pheno_all, mvals_all)
cpgs = mvals_all.columns.values
feats = pheno_all.columns.values
df_all = pd.merge(pheno_all, mvals_all, left_index=True, right_index=True)

df_train_val = df_all.loc[df_all['Dataset'].isin(datasets_train_val), :]
pheno_train_val = df_train_val.loc[:, feats]
mvals_train_val = df_train_val.loc[:, cpgs]
pheno_train_val.to_pickle(f"{path_wd}/all_in_one/pheno_train_val.pkl")
pheno_train_val.to_excel(f"{path_wd}/all_in_one/pheno_train_val.xlsx", index=True)
mvals_train_val.to_pickle(f"{path_wd}/all_in_one/mvals_train_val.pkl")

df_test = df_all.loc[df_all['Dataset'].isin(datasets_test), :]
pheno_test = df_test.loc[:, feats]
mvals_test = df_test.loc[:, cpgs]
pheno_test.to_pickle(f"{path_wd}/all_in_one/pheno_test.pkl")
pheno_test.to_excel(f"{path_wd}/all_in_one/pheno_test.xlsx", index=True)
mvals_test.to_pickle(f"{path_wd}/all_in_one/mvals_test.pkl")

cpgs_metrics_df = KW_Control(datasets_train_val, manifest, df_train_val, cpgs, f"{path_wd}/all_in_one/KW", "M value")
for cpg_id, cpg in enumerate(tqdm(cpgs)):
    cpgs_metrics_df.loc[cpg, "mean"] = df_train_val[cpg].mean()
    cpgs_metrics_df.loc[cpg, "median"] = df_train_val[cpg].median()
cpgs_metrics_df.to_excel(f"{path_wd}/all_in_one/cpgs/{cpgs_metrics_df.shape[0]}.xlsx", index=True)
