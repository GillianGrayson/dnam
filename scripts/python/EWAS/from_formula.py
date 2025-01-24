import pandas as pd
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.betas import betas_drop_na
from tqdm import tqdm
from scripts.python.EWAS.routines.correction import correct_pvalues
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.layout import add_layout
import numpy as np
from pathlib import Path
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_default_statuses_ids, get_default_statuses, get_sex_dict


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
datasets = ["GSE53740"]

is_rerun = True
num_cpgs_to_plot = 10

for dataset in datasets:
    print(dataset)
    platform = datasets_info.loc[dataset, 'platform']
    manifest = get_manifest(platform)

    statuses = get_default_statuses(dataset)
    status_col = get_column_name(dataset, 'Status').replace(' ', '_')
    statuses_ids = get_default_statuses_ids(dataset)
    status_dict = get_status_dict(dataset)
    status_passed_fields = get_passed_fields(status_dict, statuses)

    age_col = get_column_name(dataset, 'Age').replace(' ', '_')

    sex_col = get_column_name(dataset, 'Sex').replace(' ', '_')
    sex_dict = get_sex_dict(dataset)

    status_vals = sorted([x.column for x in status_passed_fields])
    sex_vals = sorted(list(sex_dict.values()))

    dnam_acc_type = 'DNAmGrimAgeAcc'

    formula = f"{age_col} + C({status_col})" # f"C({status_col})"
    terms = [f"{age_col}", f"C({status_col})[T.{status_vals[-1]}]"] # [f"C({status_col})[T.{status_vals[-1]}]"]
    aim = f"Age_Status" # f"Status"

    continuous_vars = {'Age': age_col, dnam_acc_type: dnam_acc_type}
    categorical_vars = {
        status_col: [x.column for x in status_passed_fields],
        sex_col: [sex_dict[x] for x in sex_dict]
    }
    pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
    pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
    betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
    betas = betas_drop_na(betas)
    df = pd.merge(pheno, betas, left_index=True, right_index=True)

    path_save = f"{path}/{platform}/{dataset}/EWAS/from_formula/{aim}"
    Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

    cpgs = betas.columns.values

    if is_rerun:
        result = {'CpG': cpgs}
        result['Gene'] = np.zeros(len(cpgs), dtype=object)
        metrics = ['R2', 'R2_adj']
        for m in metrics:
            result[m] = np.zeros(len(cpgs))
        for t in terms:
            result[f"{t}_pvalue"] = np.zeros(len(cpgs))

        for cpg_id, cpg in tqdm(enumerate(cpgs), desc='from_formula', total=len(cpgs)):
            result['Gene'][cpg_id] = manifest.loc[cpg, 'Gene']
            reg = smf.ols(formula=f"{cpg} ~ {formula}", data=df).fit()
            pvalues = dict(reg.pvalues)
            result['R2'][cpg_id] = reg.rsquared
            result['R2_adj'][cpg_id] = reg.rsquared_adj
            for t in terms:
                result[f"{t}_pvalue"][cpg_id] = pvalues[t]

        result = correct_pvalues(result, [f"{t}_pvalue" for t in terms])
        result = pd.DataFrame(result)
        result.set_index("CpG", inplace=True)
        result.sort_values([f"{t}_pvalue" for t in terms], ascending=[True] * len(terms), inplace=True)
        result.to_excel(f"{path_save}/table.xlsx", index=True)
    else:
        result = pd.read_excel(f"{path_save}/table.xlsx", index_col="CpG")

    result = result.head(num_cpgs_to_plot)
    for cpg_id, (cpg, row) in enumerate(result.iterrows()):
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
                Path(f"{path_save}/figs/{name_cont}_{feat}").mkdir(parents=True, exist_ok=True)
                save_figure(fig, f"{path_save}/figs/{name_cont}_{feat}/{cpg_id}_{cpg}")
