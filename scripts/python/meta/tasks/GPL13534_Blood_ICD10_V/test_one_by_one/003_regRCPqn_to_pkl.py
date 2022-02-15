import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np


path = f"E:/YandexDisk/Work/pydnameth/datasets"
task_name = f"GPL13534_Blood_ICD10-V"
path_wd = path_save = f"{path}/meta/tasks/{task_name}/test"

manifest = get_manifest('GPL13534')

test_datasets = {
    'GSE116379': ['Control', 'Schizophrenia'],
    'GSE113725': ['Control', 'Depression'],
    'GSE41169': ['Control', 'Schizophrenia'],
    'GSE116378': ['Control', 'Schizophrenia'],
}

for d_id, dataset in enumerate(test_datasets):
    print(dataset)
    mvals = pd.read_csv(f"{path_wd}/mvals_{dataset}_regRCPqn.txt", delimiter="\t", index_col='ID_REF')
    mvals = mvals.astype('float32')
    mvals = mvals.T
    mvals.index.name = "subject_id"
    print(mvals.shape)
    print(f"Number of inf values in mvals: {np.isinf(mvals).values.sum()}")
    cpgs_with_inf_mvals = mvals.columns.to_series()[np.isinf(mvals).any()]
    print(f"Number of CpGs with inf in mvals: {len(cpgs_with_inf_mvals)}")
    mvals.to_pickle(f"{path_wd}/mvals_{dataset}_regRCPqn.pkl")
