import pandas as pd
import plotly.graph_objects as go
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.histogram import add_histogram_trace
from scripts.python.routines.plot.layout import add_layout
import os
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_default_statuses, get_default_statuses_ids, get_status_dict, get_sex_dict

path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
datasets = ["GSE53740"]

for d_id, dataset in enumerate(datasets):
    print(dataset)
    platform = datasets_info.loc[dataset, 'platform']
    manifest = get_manifest(platform)

    save_path = f"{path}/{platform}/{dataset}/pheno/distribution"
    fig_path = f"{save_path}/figs"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    statuses = get_default_statuses(dataset)
    status_col = get_column_name(dataset, 'Status').replace(' ', '_')
    statuses_ids = get_default_statuses_ids(dataset)
    status_dict = get_status_dict(dataset)
    status_passed_fields = get_passed_fields(status_dict, statuses)

    status_1_cols = [status_dict['Control'][x].column for x in statuses_ids['Control']]
    status_1_label = ', '.join([status_dict['Control'][x].label for x in statuses_ids['Control']])
    status_2_cols = [status_dict['Case'][x].column for x in statuses_ids['Case']]
    status_2_label = ', '.join([status_dict['Case'][x].label for x in statuses_ids['Case']])

    age_col = get_column_name(dataset, 'Age').replace(' ', '_')

    sex_col = get_column_name(dataset, 'Sex').replace(' ', '_')
    sex_dict = get_sex_dict(dataset)

    continuous_vars = {'Age': age_col}
    categorical_vars = {
        status_col: [x.column for x in status_passed_fields],
        sex_col: [sex_dict[x] for x in sex_dict]
    }
    pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
    df = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)

    df_1 = df.loc[df[status_col].isin(status_1_cols), :]
    df_2 = df.loc[df[status_col].isin(status_2_cols), :]

    fig = go.Figure()
    add_histogram_trace(fig, df_1[age_col].values, f"{status_1_label} ({df_1.shape[0]})", 5.0)
    add_histogram_trace(fig, df_2[age_col].values, f"{status_2_label} ({df_2.shape[0]})", 5.0)
    # add_histogram_trace(fig, df_1[age_col].values, f"M ({df_1.shape[0]})", 5.0)
    # add_histogram_trace(fig, df_2[age_col].values, f"F ({df_2.shape[0]})", 5.0)
    add_layout(fig, "Age", "Count", "")
    fig.update_layout(colorway = ['blue', 'red'], barmode = 'overlay')
    #fig.update_layout(colorway = ['#19D3F3', '#FF6692'], barmode = 'overlay')
    save_figure(fig, f"{fig_path}/histogram_Age_Status")
