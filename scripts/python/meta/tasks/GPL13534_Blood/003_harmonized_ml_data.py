import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
import pathlib
from scripts.python.meta.tasks.GPL13534_Blood.routines import perform_test_for_controls
from tqdm import tqdm
import numpy as np
from scripts.python.routines.plot.layout import add_layout, get_axis
from scripts.python.routines.plot.save import save_figure
import plotly.graph_objects as go
import plotly.express as px


thld_above = 0.5
thld_below = 0.05

path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
manifest = get_manifest('GPL13534')

# disease = "Schizophrenia"
# dataset_statuses = {
#     'GSE84727': ['Control', 'Schizophrenia'],
#     'GSE80417': ['Control', 'Schizophrenia'],
#     'GSE152027': ['Control', 'Schizophrenia'],
#     'GSE116379': ['Control', 'Schizophrenia'],
#     'GSE41169': ['Control', 'Schizophrenia'],
#     'GSE116378': ['Control', 'Schizophrenia'],
#     'GSE87571': ['Control'],
# }
# datasets_trn_val = ['GSE84727', 'GSE80417']
# datasets_tst = ['GSE152027', 'GSE116379', 'GSE41169', 'GSE116378', 'GSE87571']

disease = "Parkinson"
dataset_statuses = {
    'GSE145361': ['Control', 'Parkinson'],
    'GSE111629': ['Control', 'Parkinson'],
    'GSE72774': ['Control', 'Parkinson'],
}
datasets_trn_val = ['GSE145361', 'GSE111629']
datasets_tst = ['GSE72774']

task_name = f"to_delete_checking_Parkinson_with_preprocessed_not_idat/{disease}"
path_wd = f"{path}/meta/tasks/{task_name}"
pathlib.Path(f"{path_wd}/harmonized/cpgs/figs").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_wd}/harmonized/cpgs/diffs").mkdir(parents=True, exist_ok=True)

# Train/Val data =======================================================================================================
pheno_trn_val = pd.DataFrame()
pheno_trn_val.index.name = 'subject_id'
mvals_trn_val = pd.DataFrame()
origin_df = pd.DataFrame()
for d_id, dataset in enumerate(datasets_trn_val):
    print(dataset)
    pheno_i = pd.read_pickle(f"{path_wd}/origin/pheno_trn_val_{dataset}.pkl")
    pheno_cols = pheno_i.columns.values
    mvals_i = pd.read_csv(f"{path_wd}/harmonized/r/mvalsT_trn_val_{dataset}_regRCPqn.txt", delimiter="\t", index_col='ID_REF')
    mvals_i = mvals_i.T
    mvals_cols = mvals_i.columns.values
    df_i = pd.merge(pheno_i, mvals_i, left_index=True, right_index=True)
    pheno_i = df_i.loc[:, pheno_cols]
    mvals_i = df_i.loc[:, mvals_cols]

    print(f"pheno_i shape: {pheno_i.shape}")
    print(f"mvals_i shape: {mvals_i.shape}")

    pheno_trn_val = pheno_trn_val.append(pheno_i, verify_integrity=True)

    mvals_i = mvals_i.T
    if d_id == 0:
        mvals_trn_val = mvals_i
    else:
        mvals_trn_val = mvals_trn_val.merge(mvals_i, how='inner', left_index=True, right_index=True)

    pheno_origin = pd.read_pickle(f"{path_wd}/origin/pheno_trn_val_{dataset}.pkl")
    mvals_origin = pd.read_pickle(f"{path_wd}/origin/mvalsT_trn_val_{dataset}.pkl")
    mvals_origin = mvals_origin.T
    origin_df_i = pd.merge(pheno_origin, mvals_origin, left_index=True, right_index=True)
    origin_df = origin_df.append(origin_df_i, verify_integrity=True)

mvals_trn_val = mvals_trn_val.T
mvals_trn_val.index.name = "subject_id"
mvals_trn_val = mvals_trn_val.astype('float32')
print(f"Number of total subjects: {mvals_trn_val.shape[0]}")
print(f"Number of total CpGs: {mvals_trn_val.shape[1]}")
pheno_trn_val, mvals_trn_val = get_pheno_betas_with_common_subjects(pheno_trn_val, mvals_trn_val)
feats = pheno_trn_val.columns.values
cpgs = mvals_trn_val.columns.values
df_trn_val = pd.merge(pheno_trn_val, mvals_trn_val, left_index=True, right_index=True)
pheno_trn_val = df_trn_val.loc[:, feats]
mvals_trn_val = df_trn_val.loc[:, cpgs]
df_trn_val.to_pickle(f"{path_wd}/harmonized/data_trn_val.pkl")
pheno_trn_val.to_excel(f"{path_wd}/harmonized/pheno_trn_val.xlsx", index=True)

# Check harmonization ==================================================================================================
if origin_df.shape != df_trn_val.shape:
    raise ValueError(f"Wrong shape")
if not origin_df.index.equals(df_trn_val.index):
    raise ValueError(f"Wrong indexes")

cpgs_metrics_harmonized_df = perform_test_for_controls(datasets_trn_val, manifest, df_trn_val, cpgs, f"{path_wd}/harmonized/cpgs/figs", "M value")
for cpg_id, cpg in enumerate(tqdm(cpgs)):
    cpgs_metrics_harmonized_df.loc[cpg, "mean"] = df_trn_val[cpg].mean()
    cpgs_metrics_harmonized_df.loc[cpg, "median"] = df_trn_val[cpg].median()
cpgs_metrics_harmonized_df.to_excel(f"{path_wd}/harmonized/cpgs/{cpgs_metrics_harmonized_df.shape[0]}.xlsx", index=True)
cpgs_metrics_origin_df = pd.read_excel(f"{path_wd}/origin/cpgs/{cpgs_metrics_harmonized_df.shape[0]}.xlsx", index_col="CpG")
cpgs_info = cpgs_metrics_origin_df.merge(cpgs_metrics_harmonized_df, left_index=True, right_index=True, suffixes=('_origin', '_harmonized'))
cpgs_info['log_diff_harmonized'] = np.log10(cpgs_info.loc[:, 'pval_fdr_bh_harmonized'].values) - np.log10(cpgs_info.loc[:, 'pval_fdr_bh_origin'].values)
cpgs_changed = cpgs_info.loc[(cpgs_info['pval_fdr_bh_harmonized'] > thld_above) & (cpgs_info['pval_fdr_bh_origin'] < thld_below), :]
cpgs_changed.sort_values(['log_diff_harmonized'], ascending=[False], inplace=True)
cpgs_changed.to_excel(f"{path_wd}/harmonized/cpgs/cpgs_changed_{thld_above}_{thld_below}.xlsx", index=True)

# Plot harmonization ===================================================================================================
cpgs_to_plot_df = cpgs_changed.head(20)
for cpg_id, (cpg, row) in enumerate(cpgs_to_plot_df.iterrows()):
    dist_num_bins = 25
    pval = row['pval_fdr_bh_origin']
    gene = manifest.at[cpg, 'Gene']
    fig = go.Figure()
    for dataset in datasets_trn_val:
        vals_i = origin_df.loc[(origin_df['Status'] == 'Control') & (origin_df['Dataset'] == dataset), cpg].values
        fig.add_trace(
            go.Violin(
                y=vals_i,
                name=dataset,
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
                marker=dict(line=dict(width=0.3), opacity=0.8),
                points='all',
                bandwidth=np.ptp(vals_i) / dist_num_bins,
                opacity=0.8
            )
        )
    add_layout(fig, "", "M value", f"{cpg} ({gene})<br>p-value: {pval:0.2e}")
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_xaxes(tickfont_size=15)
    fig.update_layout({'colorway': px.colors.qualitative.Set1})
    fig.update_layout(
        margin=go.layout.Margin(
            l=110,
            r=20,
            b=50,
            t=80,
            pad=0
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.25,
            xanchor="center",
            x=0.5
        )
    )
    save_figure(fig, f"{path_wd}/harmonized/cpgs/diffs/{cpg_id:03d}_{cpg}_origin")

    pval = row['pval_fdr_bh_harmonized']
    gene = manifest.at[cpg, 'Gene']
    fig = go.Figure()
    for dataset in datasets_trn_val:
        vals_i = df_trn_val.loc[(df_trn_val['Status'] == 'Control') & (df_trn_val['Dataset'] == dataset), cpg].values
        fig.add_trace(
            go.Violin(
                y=vals_i,
                name=dataset,
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
                marker=dict(line=dict(width=0.3), opacity=0.8),
                points='all',
                bandwidth=np.ptp(vals_i) / dist_num_bins,
                opacity=0.8
            )
        )
    add_layout(fig, "", "M value", f"{cpg} ({gene})<br>p-value: {pval:0.2e}")
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_xaxes(tickfont_size=15)
    fig.update_layout({'colorway': px.colors.qualitative.Set1})
    fig.update_layout(
        margin=go.layout.Margin(
            l=110,
            r=20,
            b=50,
            t=80,
            pad=0
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.25,
            xanchor="center",
            x=0.5
        )
    )
    save_figure(fig, f"{path_wd}/harmonized/cpgs/diffs/{cpg_id:03d}_{cpg}_harmonized")

# Test data ============================================================================================================
for d_id, dataset in enumerate(datasets_tst):
    print(dataset)
    mvals = pd.read_csv(f"{path_wd}/harmonized/r/mvalsT_tst_{dataset}_regRCPqn.txt", delimiter="\t", index_col='ID_REF')
    mvals = mvals.T
    mvals.index.name = "subject_id"
    mvals = mvals.astype('float32')
    pheno = pd.read_pickle(f"{path_wd}/origin/pheno_tst_{dataset}.pkl")
    print(f"Number of total subjects: {pheno.shape[0]}")
    print(f"Number of total CpGs: {mvals.shape[1]}")
    cpgs = mvals.columns.values
    feats = pheno.columns.values
    df = pd.merge(pheno, mvals, left_index=True, right_index=True)
    df.to_pickle(f"{path_wd}/harmonized/data_tst_{dataset}.pkl")
    pheno_test = df.loc[:, feats]
    pheno_test.to_excel(f"{path_wd}/harmonized/pheno_tst_{dataset}.xlsx", index=True)

