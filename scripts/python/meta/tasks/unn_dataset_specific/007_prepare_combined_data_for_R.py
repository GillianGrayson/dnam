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
from scripts.python.routines.manifest import get_genes_list
from sklearn.feature_selection import VarianceThreshold
from scripts.python.routines.plot.histogram import add_histogram_trace
import statsmodels.formula.api as smf
import pyreadr


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

gse_dataset = 'GSE87571'

path_save = f"{path}/meta/tasks/unn_dataset_specific/007_prepare_combined_data_for_R/{gse_dataset}"
Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

target_features = ['Status', 'Age', 'Sex']
calc_features = ["DNAmAge", "DNAmAgeHannum", "DNAmPhenoAge", "DNAmGrimAge"]
thlds_mv = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
thlds_var = [0.0, 0.01, 0.005, 0.001]
num_cpgs_to_plot = 10

pheno_all = pd.DataFrame(columns=target_features + ['Dataset'] + calc_features)
pheno_all.index.name = 'subject_id'

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

    pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
    pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
    betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
    betas = betas_drop_na(betas)
    df = pd.merge(pheno, betas, left_index=True, right_index=True)
    df = df.loc[df[status_col].isin(controls_status_vals), :]

    status_dict_inverse = dict((x.column, x.label) for x in status_passed_fields)
    df[status_col].replace(status_dict_inverse, inplace=True)
    df.rename(columns={status_col: 'Status'}, inplace=True)
    sex_dict_inverse = {v: k for k, v in sex_dict.items()}
    df[sex_col].replace(sex_dict_inverse, inplace=True)
    df.rename(columns={sex_col: 'Sex'}, inplace=True)
    df.rename(columns={age_col: 'Age'}, inplace=True)
    df.loc[:, 'Dataset'] = dataset
    pheno = df.loc[:, target_features + ['Dataset'] + calc_features]
    pheno_all = pheno_all.append(pheno, verify_integrity=True)

    cpgs = betas.columns.values
    betas = df[cpgs].T

    if d_id == 0:
        betas_all = betas
    else:
        betas_all = betas_all.merge(betas, how='inner', left_index=True, right_index=True)

print(f"Number of remaining subjects: {pheno_all.shape[0]}")
betas_all = betas_all.T
betas_all.index.name = "subject_id"

pheno_all, betas_all = get_pheno_betas_with_common_subjects(pheno_all, betas_all)

betas_all = betas_all.T
betas_all.index.name = "CpG"

pheno_all.to_excel(f"{path_save}/pheno.xlsx", index=True)
#betas_all.to_csv(f"{path_save}/betas.csv", index=True, index_label="CpG")
pyreadr.write_rdata(f"{path_save}/betas.RData", betas_all, df_name="tmpNorm")
