import copy

import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
import os
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_statuses_datasets_dict
from sklearn.feature_selection import VarianceThreshold
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.betas import betas_drop_na
import hashlib
import pickle
import json
import pathlib
from scipy.stats import chi2_contingency
from scipy.stats import norm
import math


dataset = "GSE52588"

path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

path_save = f"{path}/{platform}/{dataset}/special/003_chi_square"
pathlib.Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

df = pd.read_excel(f"{path}/{platform}/{dataset}/special/002_test_tabnet/chi_square.xlsx", index_col="Group")

chi2, p, dof, ex = chi2_contingency(df, correction=False)

ololo = 1