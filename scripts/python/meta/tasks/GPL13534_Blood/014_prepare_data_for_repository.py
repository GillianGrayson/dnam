import numpy as np
import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
import pathlib
from tqdm import tqdm
from impyute.imputation.cs import fast_knn, mean, median, random, mice, mode, em


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
manifest = get_manifest('GPL13534')

disease = "Schizophrenia" #"Parkinson"
data_type = "non_harmonized"
datasets_tst = ['GSE152027', 'GSE116379'] #['GSE72774']

path_load = f"{path}/meta/tasks/GPL13534_Blood/{disease}/{data_type}"
path_save = f"{path}/meta/tasks/GPL13534_Blood/release/{disease}/{data_type}"

cpgs = pd.read_excel(f"{path_save}/features.xlsx").loc[:, 'feature'].values
features = ["Status", "Dataset"] + list(cpgs)
df = pd.read_pickle(f"{path_load}/data_trn_val.pkl")
df = df.loc[:, features]
df.insert(loc=2, column='Partition', value=["Train"]*df.shape[0])

df_train = df.copy()

for dataset in datasets_tst:
    tmp = pd.read_pickle(f"{path_load}/data_tst_{dataset}.pkl")
    lost_cpgs = list(set(cpgs) - set(tmp.columns))
    for cpg in lost_cpgs:
        tmp[cpg] = np.nan
    if len(lost_cpgs) > 0:
        print(f"There are {len(lost_cpgs)} lost CpGs")
        only_cpgs_data = pd.concat([df_train.loc[:, cpgs], tmp.loc[:, cpgs]])
        only_cpgs_data = only_cpgs_data.astype('float')
        imputed_training = fast_knn(only_cpgs_data.loc[:, cpgs].values, k=1)
        only_cpgs_data.loc[:,:] = imputed_training.astype('float32')
        tmp.loc[tmp.index.values, cpgs] = only_cpgs_data.loc[tmp.index.values, cpgs]
    tmp = tmp.loc[:, features]
    tmp.insert(loc=2, column='Partition', value=["Validation"]*tmp.shape[0])
    df = pd.concat([df, tmp])

df.to_excel(f"{path_save}/data.xlsx", index=True)
