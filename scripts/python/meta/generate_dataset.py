import copy

import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_statuses_datasets_dict
from sklearn.feature_selection import VarianceThreshold
from skfeature.function.similarity_based import lap_score
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
import plotly.express as px
from scripts.python.routines.betas import betas_drop_na


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

include_controls = True
# statuses = {
#     'Control': 0,
#     'Schizophrenia': 1,
#     'First episode psychosis': 2,
#     'Parkinson': 3,
#     'Depression': 4,
#     'Intellectual disability and congenital anomalies': 5,
#     'Frontotemporal dementia': 6,
#     'Sporadic Creutzfeldt-Jakob disease': 7,
#     'Mild cognitive impairment': 8,
#     'Alzheimer': 9,
# }
statuses = {
    'Schizophrenia': 0,
    'First episode psychosis': 1,
    'Parkinson': 2,
    'Depression': 3,
    'Intellectual disability and congenital anomalies': 4,
    'Frontotemporal dementia': 5,
    'Sporadic Creutzfeldt-Jakob disease': 6,
    'Mild cognitive impairment': 7,
    'Alzheimer': 8,
}


target_features = ['Status']
metric = 'variance' # 'list' 'variance'
thld = 0.01

statuses_datasets_dict = get_statuses_datasets_dict()
datasets = set()
for s in statuses.keys():
    if s in statuses_datasets_dict:
        for dataset in statuses_datasets_dict[s]:
            datasets.add(dataset)

folder_name = f"classes_{len(statuses)}"
path_save = f"{path}/meta/{folder_name}"
if not os.path.exists(f"{path_save}/figs"):
    os.makedirs(f"{path_save}/figs")

pheno_all = pd.DataFrame(columns=target_features + ['StatusFull', 'Dataset'])
pheno_all.index.name = 'subject_id'
for d_id, dataset in enumerate(datasets):
    print(dataset)
    platform = datasets_info.loc[dataset, 'platform']
    manifest = get_manifest(platform)

    status_col = get_column_name(dataset, 'Status').replace(' ', '_')
    status_dict = get_status_dict(dataset)
    status_passed_fields = get_passed_fields(status_dict, statuses.keys())
    continuous_vars = {}
    categorical_vars = {status_col: [x.column for x in status_passed_fields]}
    pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno.pkl")
    pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
    betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
    betas = betas_drop_na(betas)
    df = pd.merge(pheno, betas, left_index=True, right_index=True)

    pheno = df.loc[:, [status_col]]
    status_dict_inverse = dict((x.column, x.label) for x in status_passed_fields)
    pheno[status_col].replace(status_dict_inverse, inplace=True)
    pheno.rename(columns={status_col: 'Status'}, inplace=True)
    pheno['StatusFull'] = pheno['Status']
    pheno.loc[:, 'Dataset'] = dataset
    pheno_all = pheno_all.append(pheno, verify_integrity=True)

    cpgs = betas.columns.values
    betas = df[cpgs].T
    if d_id == 0:
        betas_all = betas
    else:
        betas_all = betas_all.merge(betas, how='inner', left_index=True, right_index=True)

status_counts = pheno_all['Status'].value_counts()
fig = go.Figure()
add_bar_trace(fig, x=status_counts.index.values, y=status_counts.values)
add_layout(fig, "Status", "Count", f"")
fig.update_layout({'colorway': ['red']})
save_figure(fig, f"{path_save}/figs/status_counts")
print(f"Number of remaining subjects: {pheno_all.shape[0]}")

pheno_all['Status'].replace(statuses, inplace=True)
betas_all = betas_all.T
betas_all.index.name = "subject_id"
betas_all = betas_all.astype('float32')

cpgs_metrics_dict = {'CpG': betas_all.columns.values}
vt = VarianceThreshold(thld)
vt.fit(betas_all)
vt_metrics = vt.variances_
vt_bool = vt.get_support()
cpgs_metrics_dict['variance'] = vt_metrics
cpgs_metrics_df = pd.DataFrame(cpgs_metrics_dict)
cpgs_metrics_df.set_index('CpG', inplace=True)
cpgs_metrics_df.to_excel(f"{path_save}/cpgs_metrics.xlsx", index=True)

for m in ["variance"]:
    plot = cpgs_metrics_df[m].plot.kde(ind=np.logspace(-5, 0, 501))
    plt.xlabel("Values", fontsize=15)
    plt.ylabel("PDF",fontsize=15)
    plt.xscale('log')
    plt.grid(True)
    fig = plot.get_figure()
    fig.savefig(f"{path_save}/figs/{m}.pdf")
    fig.savefig(f"{path_save}/figs/{m}.png")
    plt.close()

if metric in ["variance"]:
    cpgs = cpgs_metrics_df.index.values
    ids = np.where(cpgs_metrics_df[metric].values > thld)[0]
    cpgs_target = cpgs[ids]
    path_save = f"{path}/meta/{folder_name}/{metric}({thld})"

else:
    with open(f"cpgs.txt") as f:
        cpgs_target = f.read().splitlines()
    cpgs_target = list(set.intersection(set(betas_all.columns.values), set(cpgs_target)))
    path_save = f"{path}/meta/{folder_name}/from_{len(cpgs_target)}"

if not os.path.exists(f"{path_save}"):
    os.makedirs(f"{path_save}")
print(f"Number of remaining CpGs: {len(cpgs_target)}")

curr_manifest = manifest.loc[cpgs_target, :]
curr_manifest.to_excel(f"{path_save}/manifest.xlsx", index=True)

betas_all = betas_all.loc[:, cpgs_target]

pheno_all, betas_all = get_pheno_betas_with_common_subjects(pheno_all, betas_all)
save_pheno_betas_to_pkl(pheno_all, betas_all, f"{path_save}")
