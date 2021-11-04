import pandas as pd
from scripts.python.routines.manifest import get_manifest
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout
import statsmodels.formula.api as smf
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_default_statuses, get_default_statuses_ids, get_sex_dict
from pathlib import Path
from scripts.python.routines.betas import betas_drop_na
from scipy.stats import mannwhitneyu


dataset = "GSE53740"

path = f"E:/YandexDisk/Work/pydnameth/datasets"

datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

path_save = f"{path}/{platform}/{dataset}/EWAS/cpgs_plot/"
Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

status_col = get_column_name(dataset, 'Status').replace(' ','_')
status_dict = get_status_dict(dataset)
statuses_ids = get_default_statuses_ids(dataset)
statuses = get_default_statuses(dataset)
status_passed_fields = get_passed_fields(status_dict, statuses)
status_1_cols = [status_dict['Control'][x].column for x in statuses_ids['Control']]
status_1_label = ', '.join([status_dict['Control'][x].label for x in statuses_ids['Control']])
status_2_cols = [status_dict['Case'][x].column for x in statuses_ids['Case']]
status_2_label = ', '.join([status_dict['Case'][x].label for x in statuses_ids['Case']])

age_col = get_column_name(dataset, 'Age').replace(' ','_')

sex_col = get_column_name(dataset, 'Sex').replace(' ','_')
sex_dict = get_sex_dict(dataset)

continuous_vars = {'Age': age_col}
categorical_vars = {
    status_col: [x.column for x in status_passed_fields],
    sex_col: [sex_dict[x] for x in sex_dict]
}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
betas = betas_drop_na(betas)

df = pd.merge(pheno, betas, left_index=True, right_index=True)

with open(f"cpgs_to_plot.txt") as f:
    cpgs = f.read().splitlines()

for cpg_id, cpg in enumerate(cpgs):

    statistic, pvalue = mannwhitneyu(df.loc[df[status_col].isin(status_1_cols), cpg].values,
                                     df.loc[df[status_col].isin(status_2_cols), cpg].values)
    box = go.Figure()
    add_violin_trace(box, df.loc[df[status_col].isin(status_1_cols), cpg].values, status_1_label)
    add_violin_trace(box, df.loc[df[status_col].isin(status_2_cols), cpg].values, status_2_label)
    add_layout(box, dataset, cpg, f"p-val = {pvalue:0.4e}")
    box.update_layout({'colorway': ['blue', 'red']})
    save_figure(box, f"{path_save}/violin_{cpg_id}_{cpg}")

    for name_cont, feat_cont in continuous_vars.items():
        for feat, groups in categorical_vars.items():
            fig = go.Figure()
            for group_val in groups:
                df_curr = df.loc[df[feat] == group_val, :]
                reg = smf.ols(formula=f"{cpg} ~ {feat_cont}", data=df_curr).fit()
                add_scatter_trace(fig, df_curr[feat_cont].values, df_curr[cpg].values, group_val)
                add_scatter_trace(fig, df_curr[feat_cont].values, reg.fittedvalues.values, "", "lines")
            add_layout(fig, name_cont, 'Methylation Level', f"{cpg} ({manifest.loc[cpg, 'Gene']})")
            fig.update_layout({'colorway': ['blue', 'blue', "red", "red"]})
            Path(f"{path_save}/{name_cont}_{feat}").mkdir(parents=True, exist_ok=True)
            save_figure(fig, f"{path_save}/{name_cont}_{feat}/{cpg_id}_{cpg}")
