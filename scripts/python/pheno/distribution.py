import pandas as pd
import plotly.graph_objects as go
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.histogram import add_histogram_trace
from scripts.python.routines.plot.layout import add_layout
import os
from scripts.python.pheno.datasets.filter import filter_pheno
from scripts.python.pheno.datasets.features import get_column_name, get_status_names_dict, get_status_dict, \
    get_sex_dict

path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets = ["GSE84727", "GSE147221", "GSE125105", "GSE111629", "GSE72774"]
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

for d_id, dataset in enumerate(datasets):
    print(dataset)

    platform = datasets_info.loc[dataset, 'platform']
    manifest = get_manifest(platform)

    save_path = f"{path}/{platform}/{dataset}/pheno/distribution"
    fig_path = f"{save_path}/figs"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    status_col = get_column_name(dataset, 'Status').replace(' ','_')
    age_col = get_column_name(dataset, 'Age').replace(' ','_')
    sex_col = get_column_name(dataset, 'Sex').replace(' ','_')
    status_dict = get_status_dict(dataset)
    get_status_names = get_status_names_dict(dataset)
    sex_dict = get_sex_dict(dataset)

    continuous_vars = {'Age': age_col}
    categorical_vars = {status_col: status_dict, sex_col: sex_dict}
    pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
    pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)

    df_1 = pheno.loc[(pheno[status_col] == status_dict['Control']), :]
    df_2 = pheno.loc[(pheno[status_col] == status_dict['Case']), :]

    # df_1 = pheno.loc[(pheno[sex_col] == sex_dict['M']), :]
    # df_2 = pheno.loc[(pheno[sex_col] == sex_dict['F']), :]

    fig = go.Figure()
    add_histogram_trace(fig, df_1[age_col].values, f"{get_status_names['Control']} ({df_1.shape[0]})", 5.0)
    add_histogram_trace(fig, df_2[age_col].values, f"{get_status_names['Case']} ({df_2.shape[0]})", 5.0)
    # add_histogram_trace(fig, df_1[age_col].values, f"M ({df_1.shape[0]})", 5.0)
    # add_histogram_trace(fig, df_2[age_col].values, f"F ({df_2.shape[0]})", 5.0)
    add_layout(fig, "Age", "Count", "")
    fig.update_layout(colorway = ['blue', 'red'], barmode = 'overlay')
    #fig.update_layout(colorway = ['#19D3F3', '#FF6692'], barmode = 'overlay')
    save_figure(fig, f"{fig_path}/histogram_Age_Status")
