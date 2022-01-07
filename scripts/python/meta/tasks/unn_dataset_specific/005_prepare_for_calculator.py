import pandas as pd
import plotly.graph_objects as go
from scripts.python.preprocessing.serialization.routines.filter import get_forbidden_cpgs, manifest_filter
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
from pathlib import Path
import numpy as np


path = f"E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/unn_dataset_specific"
platform = 'GPL21145'

manifest = get_manifest(platform)

path_save = f"{path}/005_prepare_for_calculator"
Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

tissue = 'Blood WB' # "Liver" "Brain FCTX" "Blood WB"

age_col = 'Age'
sex_col = 'Sex'
sex_dict = {"F": "F", "M": "M"}

pheno = pd.read_pickle(f"{path}/004_prepare_python_data/pheno.pkl")
pheno = pheno[[age_col, sex_col]]
pheno[sex_col] = pheno[sex_col].map({sex_dict["F"]: 1, sex_dict["M"]: 0})
pheno.rename(columns={age_col: 'Age', sex_col: 'Female'}, inplace=True)
pheno["Tissue"] = tissue
pheno.to_csv(f"{path_save}/pheno.csv", na_rep="NA")

with open(f"E:/YandexDisk/Work/pydnameth/datasets/lists/cpgs/cpgs_horvath_calculator.txt") as f:
    cpgs_h = f.read().splitlines()
betas = pd.read_pickle(f"{path}/004_prepare_python_data/betas.pkl")
cpgs_na = list(set(cpgs_h) - set(betas.columns.values))
betas = betas[betas.columns.intersection(cpgs_h)]
betas[cpgs_na] = np.nan
betas = betas.T
betas.to_csv(f"{path_save}/betas.csv", na_rep="NA")
