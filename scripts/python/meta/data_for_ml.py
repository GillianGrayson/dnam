import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.layout import add_layout
import os
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.filter import filter_pheno
from scripts.python.pheno.datasets.features import get_column_name, get_status_names_dict, get_status_dict, \
    get_sex_dict
from sklearn.feature_selection import VarianceThreshold
from skfeature.function.similarity_based import lap_score
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl


platform = "GPL13534"
path = f"E:/YandexDisk/Work/pydnameth/datasets"

datasets_diseases = {
    "GSE84727": "Schizophrenia",
    "GSE147221": "Schizophrenia",
    "GSE125105": "Depression",
    "GSE111629": "Parkinson's",
    "GSE128235": "Depression",
    "GSE72774": "Parkinson's"
}

diseases_keys = {
    "Control": 0,
    "Schizophrenia": 1,
    "Depression": 2,
    "Parkinson's": 3
}

target = 'BrainDiseases'
metric = 'variance'
thld = 0.005

path_save = f"{path}/{platform}/meta/{target}"
if not os.path.exists(f"{path_save}/figs"):
    os.makedirs(f"{path_save}/figs")

manifest = get_manifest(platform)

pheno_all = pd.DataFrame(columns=['Age', 'Sex', 'Status'])
pheno_all.index.name = 'subject_id'
for d_id, (dataset, disease) in enumerate(datasets_diseases.items()):
    print(dataset)
    status_col = get_column_name(dataset, 'Status').replace(' ', '_')
    age_col = get_column_name(dataset, 'Age').replace(' ', '_')
    sex_col = get_column_name(dataset, 'Sex').replace(' ', '_')
    status_dict = get_status_dict(dataset)
    status_vals = sorted(list(status_dict.values()))
    status_names_dict = get_status_names_dict(dataset)
    sex_dict = get_sex_dict(dataset)

    continuous_vars = {'Age': age_col}
    categorical_vars = {status_col: status_dict, sex_col: sex_dict}
    pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
    pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
    betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
    na_cols = betas.columns[betas.isna().any()].tolist()
    if len(na_cols) > 0:
        print(f"CpGs with NaNs in {dataset}: {na_cols}")
        s = betas.stack(dropna=False)
        na_pairs = [list(x) for x in s.index[s.isna()]]
        print(*na_pairs, sep='\n')
    betas.dropna(axis='columns', how='any', inplace=True)
    df = pd.merge(pheno, betas, left_index=True, right_index=True)

    pheno = df[[age_col, sex_col, status_col]]
    status_dict[datasets_diseases[dataset]] = status_dict.pop("Case")
    status_dict_inverse = dict((v, k) for k, v in status_dict.items())
    pheno[status_col].replace(status_dict_inverse, inplace=True)
    sex_dict_inverse = dict((v, k) for k, v in sex_dict.items())
    pheno[sex_col].replace(sex_dict_inverse, inplace=True)
    pheno.rename(columns={age_col: 'Age', sex_col: 'Sex', status_col: 'Status'}, inplace=True)
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

pheno_all['Status'].replace(diseases_keys, inplace=True)

betas_all = betas_all.T
betas_all.index.name = "subject_id"
betas_all = betas_all.astype('float32')

cpgs_metrics_dict = {'CpG': betas_all.columns.values}
vt = VarianceThreshold(thld)
vt.fit(betas_all)
vt_metrics = vt.variances_
vt_bool = vt.get_support()
cpgs_metrics_dict['variance'] = vt_metrics
lap_metrics = lap_score.lap_score(betas_all.to_numpy())
lap_idx = lap_score.feature_ranking(lap_metrics)
cpgs_metrics_dict["lap_score"] = lap_metrics
cpgs_metrics_df = pd.DataFrame(cpgs_metrics_dict)
cpgs_metrics_df.set_index('CpG', inplace=True)
cpgs_metrics_df.to_excel(f"{path_save}/cpgs_metrics.xlsx", index=True)

for m in ["variance", "lap_score"]:
    plot = cpgs_metrics_df[m].plot.kde(ind=np.logspace(-5, 0, 501))
    plt.xlabel("Values", fontsize=15)
    plt.ylabel("PDF",fontsize=15)
    plt.xscale('log')
    plt.grid(True)
    fig = plot.get_figure()
    fig.savefig(f"{path_save}/figs/{m}.pdf")
    fig.savefig(f"{path_save}/figs/{m}.png")
    plt.close()

path_save = f"{path}/{platform}/meta/{target}/{metric}({thld})"
if not os.path.exists(f"{path_save}"):
    os.makedirs(f"{path_save}")

cpgs = cpgs_metrics_df.index.values
ids = np.where(cpgs_metrics_df[metric].values > thld)[0]
filtered_cpgs = cpgs[ids]
print(f"Number of remaining CpGs: {len(filtered_cpgs)}")

curr_manifest = manifest.loc[filtered_cpgs, :]
curr_manifest.to_excel(f"{path_save}/manifest.xlsx", index=True)

betas_all = betas_all.loc[:, filtered_cpgs]

pheno_all, betas_all = get_pheno_betas_with_common_subjects(pheno_all, betas_all)
save_pheno_betas_to_pkl(pheno_all, betas_all, f"{path_save}")