import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
from scripts.python.routines.manifest import get_manifest
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout
import os
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_default_statuses, get_default_statuses_ids, get_status_dict, get_sex_dict


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
datasets = ["GSE53740"]

is_update = True

for dataset in datasets:
    print(dataset)
    platform = datasets_info.loc[dataset, 'platform']
    manifest = get_manifest(platform)

    save_path = f"{path}/{platform}/{dataset}/pheno/acceleration"
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

    y_feats = ["DNAmAge", "DNAmAgeHannum", "DNAmPhenoAge", "DNAmGrimAge"]
    res_names = ["DNAmAgeAcc", "DNAmAgeHannumAcc", "DNAmPhenoAgeAcc", "DNAmGrimAgeAcc"]

    continuous_vars = {'Age': age_col}
    categorical_vars = {
        status_col: [x.column for x in status_passed_fields],
        sex_col: [sex_dict[x] for x in sex_dict]
    }
    pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
    df = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)

    df_1 = df.loc[df[status_col].isin(status_1_cols), :]
    df_2 = df.loc[df[status_col].isin(status_2_cols), :]

    metrics = ['R2', 'R2_adj', 'MW_statistic', 'MW_pvalue']
    res_dict = {"metric": y_feats}
    for m in metrics:
        res_dict[m] = np.zeros(len(y_feats))

    for y_id, y in enumerate(y_feats):
        formula = f"{y} ~ {age_col}"
        reg = smf.ols(formula=formula, data=df_1).fit()
        res_dict['R2'][y_id] = reg.rsquared
        res_dict['R2_adj'][y_id] = reg.rsquared_adj

        if is_update:
            pheno[res_names[y_id]] = pheno[y] - reg.predict(pheno)
            df[res_names[y_id]] = df[y] - reg.predict(df)
            df_1 = df.loc[df[status_col].isin(status_1_cols), :]
            df_2 = df.loc[df[status_col].isin(status_2_cols), :]

        scatter = go.Figure()
        add_scatter_trace(scatter, df_1[age_col].values, df_1[y].values, status_1_label)
        add_scatter_trace(scatter, df_1[age_col].values, reg.fittedvalues.values, "", "lines")
        add_scatter_trace(scatter, df_2[age_col].values, df_2[y].values, status_2_label)
        add_layout(scatter, "Age", y, "")
        scatter.update_layout({'colorway': ['blue', 'blue', 'red']})
        save_figure(scatter, f"{fig_path}/scatter_Age_{y}")

        statistic, pvalue = mannwhitneyu(df_1[res_names[y_id]].values, df_2[res_names[y_id]].values)

        res_dict['MW_statistic'][y_id] = statistic
        res_dict['MW_pvalue'][y_id] = pvalue

        vio = go.Figure()
        add_violin_trace(vio, df_1[res_names[y_id]].values, status_1_label)
        add_violin_trace(vio, df_2[res_names[y_id]].values, status_2_label)
        add_layout(vio, "", res_names[y_id], f"{res_names[y_id]}: {pvalue:0.4e}")
        vio.update_layout({'colorway': ['blue', 'red']})
        save_figure(vio, f"{fig_path}/vio_{res_names[y_id]}")

    res_df = pd.DataFrame(res_dict)
    res_df.set_index("metric", inplace=True)
    res_df.to_excel(f"{save_path}/table.xlsx", index=True)

    pheno.to_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
    pheno.to_excel(f"{path}/{platform}/{dataset}/pheno_xtd.xlsx", index=True)

