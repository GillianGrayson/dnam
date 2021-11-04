import pandas as pd
from scripts.python.routines.manifest import get_manifest
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from scripts.python.EWAS.routines.correction import correct_pvalues
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
from scripts.python.routines.plot.box import add_box_trace
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_default_statuses, get_default_statuses_ids, get_sex_dict
from scripts.python.routines.betas import betas_drop_na
from pathlib import Path


path = f"E:/YandexDisk/Work/pydnameth/datasets"
dataset = "GSE53740"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)

is_rerun = True
num_cpgs_to_plot = 10

path_save = f"{path}/{platform}/{dataset}/EWAS/mann_whitney_u_test"
Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

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
df_1 = df.loc[df[status_col].isin(status_1_cols), :]
df_2 = df.loc[df[status_col].isin(status_2_cols), :]

cpgs = betas.columns.values

if is_rerun:
    result = {'CpG': cpgs}
    result['Gene'] = np.zeros(len(cpgs), dtype=object)
    metrics = ['statistic', 'pval']
    for m in metrics:
        result[m] = np.zeros(len(cpgs))

    for cpg_id, cpg in tqdm(enumerate(cpgs), desc='Mann-Whitney U test', total=len(cpgs)):
        result['Gene'][cpg_id] = manifest.loc[cpg, 'Gene']
        data_1 = df_1[cpg].values
        data_2 = df_2[cpg].values
        statistic, pvalue = mannwhitneyu(data_1, data_2)
        result['statistic'][cpg_id] = statistic
        result['pval'][cpg_id] = pvalue

    result = correct_pvalues(result, ['pval'])
    result = pd.DataFrame(result)
    result.set_index("CpG", inplace=True)
    result.sort_values(['pval'], ascending=[True], inplace=True)
    result.to_excel(f"{path_save}/table.xlsx", index=True)
else:
    result = pd.read_excel(f"{path_save}/table.xlsx", index_col="CpG")

result = result.head(num_cpgs_to_plot)
for cpg_id, (cpg, row) in enumerate(result.iterrows()):
    fig = go.Figure()
    add_box_trace(fig, df_1[cpg].values, status_1_label)
    add_box_trace(fig, df_2[cpg].values, status_2_label)
    add_layout(fig, '', "Methylation Level", f"{cpg} ({manifest.loc[cpg, 'Gene']}): {row['pval']:0.4e}")
    fig.update_layout({'colorway': ['blue', "red"]})
    save_figure(fig, f"{path_save}/figs/{cpg_id}_{cpg}")
