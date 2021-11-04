import pandas as pd
from scripts.python.routines.manifest import get_manifest
from tqdm import tqdm
from scripts.python.EWAS.routines.correction import correct_pvalues
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.layout import add_layout
import numpy as np
from pingouin import ancova
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_default_statuses, get_default_statuses_ids, get_sex_dict
from pathlib import Path
from scripts.python.routines.betas import betas_drop_na


platform = "GPL13534"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
datasets = ["GSE53740"]

is_rerun = True
num_cpgs_to_plot = 10

for dataset in datasets:
    print(dataset)
    platform = datasets_info.loc[dataset, 'platform']
    manifest = get_manifest(platform)

    status_col = get_column_name(dataset, 'Status').replace(' ', '_')
    status_dict = get_status_dict(dataset)
    statuses_ids = get_default_statuses_ids(dataset)
    statuses = get_default_statuses(dataset)
    status_passed_fields = get_passed_fields(status_dict, statuses)
    status_1_cols = [status_dict['Control'][x].column for x in statuses_ids['Control']]
    status_1_label = ', '.join([status_dict['Control'][x].label for x in statuses_ids['Control']])
    status_2_cols = [status_dict['Case'][x].column for x in statuses_ids['Case']]
    status_2_label = ', '.join([status_dict['Case'][x].label for x in statuses_ids['Case']])

    age_col = get_column_name(dataset, 'Age').replace(' ', '_')

    sex_col = get_column_name(dataset, 'Sex').replace(' ', '_')
    sex_dict = get_sex_dict(dataset)

    terms = [status_col, age_col]
    aim = f"Age_Status"

    path_save = f"{path}/{platform}/{dataset}/EWAS/ancova/{aim}"
    Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

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

    cpgs = betas.columns.values

    if is_rerun:
        result = {'CpG': cpgs}
        result['Gene'] = np.zeros(len(cpgs), dtype=object)
        for t in terms:
            result[f"{t}_pval"] = np.zeros(len(cpgs))

        for cpg_id, cpg in tqdm(enumerate(cpgs), desc='from_formula', total=len(cpgs)):
            result['Gene'][cpg_id] = manifest.loc[cpg, 'Gene']
            res = ancova(data=df, dv=cpg, covar=age_col, between=status_col)
            for t in terms:
                result[f"{t}_pval"][cpg_id] = res.loc[res['Source'] == t, 'p-unc'].values[0]

        result = correct_pvalues(result, [f"{t}_pval" for t in terms])
        result = pd.DataFrame(result)
        result.set_index("CpG", inplace=True)
        result.sort_values([f"{t}_pval" for t in terms], ascending=[True] * len(terms), inplace=True)
        result.to_excel(f"{path_save}/table.xlsx", index=True)
    else:
        result = pd.read_excel(f"{path_save}/table.xlsx", index_col="CpG")

    result = result.head(num_cpgs_to_plot)
    for cpg_id, (cpg, row) in enumerate(result.iterrows()):
        fig = go.Figure()
        add_scatter_trace(fig,  df.loc[df[status_col].isin(status_1_cols), age_col].values, df.loc[df[status_col].isin(status_1_cols), cpg].values, status_1_label)
        add_scatter_trace(fig, df.loc[df[status_col].isin(status_2_cols), age_col].values, df.loc[df[status_col].isin(status_2_cols), cpg].values, status_2_label)
        add_layout(fig, "Age", 'Methylation Level', f"{cpg} ({manifest.loc[cpg, 'Gene']})")
        fig.update_layout({'colorway': ['blue', "red"]})
        save_figure(fig, f"{path_save}/figs/{cpg_id}_{cpg}")
