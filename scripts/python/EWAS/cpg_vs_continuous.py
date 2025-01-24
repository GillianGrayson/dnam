import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scripts.python.EWAS.routines.correction import correct_pvalues
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.layout import add_layout
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_default_statuses_ids, get_status_dict, get_default_statuses, get_sex_dict
from pathlib import Path
from scripts.python.routines.betas import betas_drop_na


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
datasets = ["GSE53740"]

is_rerun = True
num_cpgs_to_plot = 10

feats = {
    "DNAmPhenoAgeAcc": "DNAmPhenoAgeAcc",
    "DNAmGrimAgeAcc": "DNAmGrimAgeAcc"
}

for dataset in datasets:
    print(dataset)
    platform = datasets_info.loc[dataset, 'platform']
    manifest = get_manifest(platform)

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
    pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
    betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
    betas = betas_drop_na(betas)
    df = pd.merge(pheno, betas, left_index=True, right_index=True)
    df_1 = df.loc[df[status_col].isin(status_1_cols), :]
    df_2 = df.loc[df[status_col].isin(status_2_cols), :]

    path_save = f"{path}/{platform}/{dataset}/EWAS/cpg_vs_continuous/{status_1_label}"

    cpgs = betas.columns.values

    for k, v in feats.items():
        df_1_curr = df_1[df_1[k].notnull()]
        df_2_curr = df_2[df_2[k].notnull()]

        path_curr = f"{path_save}/{v}/figs"
        Path(f"{path_curr}").mkdir(parents=True, exist_ok=True)

        if is_rerun:
            result = {'CpG': cpgs}
            result['Gene'] = np.zeros(len(cpgs), dtype=object)
            metrics = ['R2', 'R2_adj', f"{v}_pval", 'pearson_r', 'pearson_pval', 'spearman_r', 'spearman_pval']
            for m in metrics:
                result[m] = np.zeros(len(cpgs))

            for cpg_id, cpg in tqdm(enumerate(cpgs), desc='Regression', total=len(cpgs)):
                result['Gene'][cpg_id] = manifest.loc[cpg, 'Gene']
                reg = smf.ols(formula=f"{cpg} ~ {k}", data=df_1_curr).fit()
                pvalues = dict(reg.pvalues)
                result['R2'][cpg_id] = reg.rsquared
                result['R2_adj'][cpg_id] = reg.rsquared_adj
                result[f"{v}_pval"][cpg_id] = pvalues[k]
                pearson_r, pearson_pval = pearsonr(df_1_curr[cpg].values, df_1_curr[k].values)
                result['pearson_r'][cpg_id] = pearson_r
                result['pearson_pval'][cpg_id] = pearson_pval
                spearman_r, spearman_pval = spearmanr(df_1_curr[cpg].values, df_1_curr[k].values)
                result['spearman_r'][cpg_id] = spearman_r
                result['spearman_pval'][cpg_id] = spearman_pval

            result = correct_pvalues(result, [f"{v}_pval", 'pearson_pval', 'spearman_pval'])
            result = pd.DataFrame(result)
            result.set_index("CpG", inplace=True)
            result.sort_values([f"{v}_pval"], ascending=[True], inplace=True)
            result.to_excel(f"{path_save}/{v}/table.xlsx", index=True)
        else:
            result = pd.read_excel(f"{path_save}/{v}/table.xlsx", index_col="CpG")

        result = result.head(num_cpgs_to_plot)
        for cpg_id, (cpg, row) in enumerate(result.iterrows()):
            reg = smf.ols(formula=f"{cpg} ~ {k}", data=df_1_curr).fit()
            fig = go.Figure()
            add_scatter_trace(fig, df_1_curr[k].values, df_1_curr[cpg].values, status_1_label)
            add_scatter_trace(fig, df_1_curr[k].values, reg.fittedvalues.values, "", "lines")
            add_scatter_trace(fig, df_2_curr[k].values, df_2_curr[cpg].values, status_2_label)
            add_layout(fig, f"{v}", 'Methylation Level', f"{cpg} ({row['Gene']})")
            fig.update_layout({'colorway': ['blue', 'blue', "red"]})
            save_figure(fig, f"{path_curr}/{cpg_id}_{cpg}")
