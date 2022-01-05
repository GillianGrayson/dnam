import pandas as pd
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_default_statuses_ids, get_status_dict, get_default_statuses, get_sex_dict
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu
from scripts.python.EWAS.routines.correction import correct_pvalues
from tqdm import tqdm
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.box import add_box_trace
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.betas import betas_drop_na
from scripts.python.routines.plot.layout import add_layout
import numpy as np
from pathlib import Path
import os
from scripts.python.routines.manifest import get_genes_list
from sklearn.feature_selection import VarianceThreshold
from scripts.python.routines.plot.histogram import add_histogram_trace
import statsmodels.formula.api as smf


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

gse_dataset = 'GSE164056'

folder_name = f"unn_dataset_specific/002_prepare_pd_for_ChAMP"
path_save = f"{path}/meta/tasks/{folder_name}/{gse_dataset}"
Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

target_features = ['Status', 'Age', 'Sex', 'Sentrix_Position', 'Sentrix_ID']

pheno_all = pd.DataFrame(columns=target_features + ['Subject_Id', 'Sample_Group'])
pheno_all.index.name = 'Sample_Name'

cohort_dict = {
    'GSEUNN': 'RU',
    'GSE164056': 'EU'
}

datasets = ['GSEUNN'] + [gse_dataset]
for d_id, dataset in enumerate(datasets):
    platform = datasets_info.loc[dataset, 'platform']
    manifest = get_manifest(platform)

    statuses = get_default_statuses(dataset)
    status_col = get_column_name(dataset, 'Status').replace(' ', '_')
    statuses_ids = get_default_statuses_ids(dataset)
    status_dict = get_status_dict(dataset)
    status_passed_fields = get_passed_fields(status_dict, statuses)
    controls_status_vals = [status_dict['Control'][x].column for x in statuses_ids['Control']]

    age_col = get_column_name(dataset, 'Age').replace(' ', '_')

    sex_col = get_column_name(dataset, 'Sex').replace(' ', '_')
    sex_dict = get_sex_dict(dataset)

    continuous_vars = {'Age': age_col}
    categorical_vars = {
        status_col: [x.column for x in status_passed_fields],
        sex_col: [sex_dict[x] for x in sex_dict]
    }

    pheno = pd.read_excel(f"{path}/{platform}/{dataset}/pheno.xlsx", index_col="Sample_Name")
    pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
    pheno = pheno.loc[pheno[status_col].isin(controls_status_vals), :]

    status_dict_inverse = dict((x.column, x.label) for x in status_passed_fields)
    pheno[status_col].replace(status_dict_inverse, inplace=True)
    pheno.rename(columns={status_col: 'Status'}, inplace=True)
    sex_dict_inverse = {v: k for k, v in sex_dict.items()}
    pheno[sex_col].replace(sex_dict_inverse, inplace=True)
    pheno.rename(columns={sex_col: 'Sex'}, inplace=True)
    pheno.rename(columns={age_col: 'Age'}, inplace=True)
    pheno.loc[:, 'Sample_Group'] = cohort_dict[dataset]
    subject_ids = [cohort_dict[dataset] + f'{i}' for i in range(pheno.shape[0])]
    pheno["Subject_Id"] = subject_ids
    pheno = pheno.loc[:, target_features + ['Subject_Id', 'Sample_Group']]


    index_old = pheno.index.to_list()
    index_new = pheno[['Sentrix_ID', 'Sentrix_Position']].apply(lambda x : '{}_{}'.format(x[0],x[1]), axis=1).values

    index_dict = dict(zip(index_old, index_new))

    for k, v in index_dict.items():
        old_file = os.path.join(f"{path_save}/raw/idat", f"{k}_Grn.idat")
        new_file = os.path.join(f"{path_save}/raw/idat", f"{v}_Grn.idat")
        os.rename(old_file, new_file)
        old_file = os.path.join(f"{path_save}/raw/idat", f"{k}_Red.idat")
        new_file = os.path.join(f"{path_save}/raw/idat", f"{v}_Red.idat")
        os.rename(old_file, new_file)

    pheno = pheno.rename(index=index_dict)

    pheno_all = pheno_all.append(pheno, verify_integrity=True)

pheno_all.to_excel(f"{path_save}/pheno.xlsx", index=True)

