import pandas as pd
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_default_statuses_ids, get_status_dict, get_default_statuses, get_sex_dict
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu
from scripts.python.EWAS.routines.correction import correct_pvalues
from tqdm import tqdm
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.box import add_box_trace
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.betas import betas_drop_na
from scripts.python.routines.plot.layout import add_layout
import numpy as np
from pathlib import Path
from scripts.python.routines.manifest import get_genes_list
from sklearn.feature_selection import VarianceThreshold
from scripts.python.routines.plot.histogram import add_histogram_trace
import statsmodels.formula.api as smf


path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

gse_dataset = 'GSE152026'

folder_name = f"unn_dataset_specific"
path_save = f"{path}/meta/tasks/{folder_name}/{gse_dataset}"
Path(f"{path_save}/figs/CpG").mkdir(parents=True, exist_ok=True)
Path(f"{path_save}/figs/DNAmAge").mkdir(parents=True, exist_ok=True)

target_features = ['Status', 'Age', 'Sex']
calc_features = ["DNAmAge", "DNAmAgeHannum", "DNAmPhenoAge", "DNAmGrimAge"]
thlds_mv = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
thlds_var = [0.0, 0.01, 0.005, 0.001]
num_cpgs_to_plot = 10

pheno_all = pd.DataFrame(columns=target_features + ['Dataset'] + calc_features)
pheno_all.index.name = 'subject_id'

datasets = ['GSEUNN'] + [gse_dataset]
for d_id, dataset in enumerate(datasets):
    platform = datasets_info.loc[dataset, 'platform']
    manifest = get_manifest(platform)

    statuses = get_default_statuses(dataset)
    status_col = get_column_name(dataset, 'Status').replace(' ', '_')
    statuses_ids = get_default_statuses_ids(dataset)
    status_dict = get_status_dict(dataset)
    status_passed_fields = get_passed_fields(status_dict, statuses)
    controls_status_vals = [status_dict['Control'][x].column for x in statuses_ids['Control']]

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
    df = df.loc[df[status_col].isin(controls_status_vals), :]

    status_dict_inverse = dict((x.column, x.label) for x in status_passed_fields)
    df[status_col].replace(status_dict_inverse, inplace=True)
    df.rename(columns={status_col: 'Status'}, inplace=True)
    sex_dict_inverse = {v: k for k, v in sex_dict.items()}
    df[sex_col].replace(sex_dict_inverse, inplace=True)
    df.rename(columns={sex_col: 'Sex'}, inplace=True)
    df.rename(columns={age_col: 'Age'}, inplace=True)
    df.loc[:, 'Dataset'] = dataset
    pheno = df.loc[:, target_features + ['Dataset'] + calc_features]
    pheno_all = pheno_all.append(pheno, verify_integrity=True)

    cpgs = betas.columns.values
    betas = df[cpgs].T

    if d_id == 0:
        betas_all = betas
    else:
        betas_all = betas_all.merge(betas, how='inner', left_index=True, right_index=True)

print(f"Number of remaining subjects: {pheno_all.shape[0]}")
betas_all = betas_all.T
betas_all.index.name = "subject_id"

pheno_all, betas_all = get_pheno_betas_with_common_subjects(pheno_all, betas_all)

platform = datasets_info.loc['GSEUNN', 'platform']
manifest = get_manifest(platform)

pheno_gse = pheno_all.loc[pheno_all['Dataset'] != 'GSEUNN', :]
pheno_unn = pheno_all.loc[pheno_all['Dataset'] == 'GSEUNN', :]

metrics = ['R2', 'R2_adj', 'mw_stat_acc', 'mw_pval_acc', 'mw_stat_diff', 'mw_pval_diff']
res_dict = {"metric": calc_features}
for m in metrics:
    res_dict[m] = np.zeros(len(calc_features))
for f_id, f in enumerate(calc_features):
    formula = f"{f} ~ Age"
    reg = smf.ols(formula=formula, data=pheno_gse).fit()
    res_dict['R2'][f_id] = reg.rsquared
    res_dict['R2_adj'][f_id] = reg.rsquared_adj

    pheno_all[f"{f}Acc"] = pheno_all[f] - reg.predict(pheno_all)
    pheno_all[f"{f}Diff"] = pheno_all[f] - pheno_all["Age"]
    pheno_gse = pheno_all.loc[pheno_all['Dataset'] != 'GSEUNN', :]
    pheno_unn = pheno_all.loc[pheno_all['Dataset'] == 'GSEUNN', :]

    scatter = go.Figure()
    add_scatter_trace(scatter, [0, 100], [0, 100], "", "lines")
    add_scatter_trace(scatter, pheno_gse["Age"].values, pheno_gse[f].values, gse_dataset)
    add_scatter_trace(scatter, pheno_gse["Age"].values, reg.fittedvalues.values, "", "lines")
    add_scatter_trace(scatter, pheno_unn["Age"].values, pheno_unn[f].values, "UNN")
    add_layout(scatter, "Age", f, "")
    scatter.update_layout({'colorway': ['black', 'blue', 'blue', 'red']})
    save_figure(scatter, f"{path_save}/figs/DNAmAge/scatter_Age_{f}")

    stat, pval = mannwhitneyu(pheno_gse[f"{f}Acc"].values, pheno_unn[f"{f}Acc"].values)
    res_dict['mw_stat_acc'][f_id] = stat
    res_dict['mw_pval_acc'][f_id] = pval
    vio = go.Figure()
    add_violin_trace(vio, pheno_gse[f"{f}Acc"].values, gse_dataset)
    add_violin_trace(vio, pheno_unn[f"{f}Acc"].values, "UNN")
    add_layout(vio, "", f"{f} acceleration", f"Mann-Whitney p-value: {pval:0.4e}")
    vio.update_layout({'colorway': ['blue', 'red']})
    save_figure(vio, f"{path_save}/figs/DNAmAge/vio_{f}Acc")

    stat, pval = mannwhitneyu(pheno_gse[f"{f}Diff"].values, pheno_unn[f"{f}Diff"].values)
    res_dict['mw_stat_diff'][f_id] = stat
    res_dict['mw_pval_diff'][f_id] = pval
    vio = go.Figure()
    add_violin_trace(vio, pheno_gse[f"{f}Diff"].values, gse_dataset)
    add_violin_trace(vio, pheno_unn[f"{f}Diff"].values, "UNN")
    add_layout(vio, "", f"{f} - Age", f"Mann-Whitney p-value: {pval:0.4e}")
    vio.update_layout({'colorway': ['blue', 'red']})
    save_figure(vio, f"{path_save}/figs/DNAmAge/vio_{f}Diff")

res_df = pd.DataFrame(res_dict)
res_df.set_index("metric", inplace=True)
res_df.to_excel(f"{path_save}/calc_features.xlsx", index=True)

pheno_all.to_pickle(f"{path_save}/pheno.pkl")
pheno_all.to_excel(f"{path_save}/pheno.xlsx", index=True)

df = pd.merge(pheno_all, betas_all, left_index=True, right_index=True)
df_unn = df.loc[df['Dataset'] == 'GSEUNN', :]
df_gse = df.loc[df['Dataset'] != 'GSEUNN', :]

fig = go.Figure()
add_histogram_trace(fig, df_unn['Age'].values, f"UNN ({df_unn.shape[0]})", 5.0)
add_histogram_trace(fig, df_gse['Age'].values, f"{gse_dataset} ({df_gse.shape[0]})", 5.0)
add_layout(fig, "Age", "Count", "")
fig.update_layout(colorway = ['blue', 'red'], barmode = 'overlay')
save_figure(fig, f"{path_save}/figs/histogram_Age")

cpgs = betas_all.columns.values
result = {'CpG': cpgs}
result['Gene'] = np.zeros(len(cpgs), dtype=object)
vt = VarianceThreshold(0.0)
vt.fit(betas_all)
vt_metrics = vt.variances_
result['variance'] = vt_metrics
metrics = ['mw_stat', 'mw_pval', 'dataset_pval']
for m in metrics:
    result[m] = np.zeros(len(cpgs))

for cpg_id, cpg in tqdm(enumerate(cpgs), desc=f'CpGs', total=len(cpgs)):
    result['Gene'][cpg_id] = manifest.loc[cpg, 'Gene']
    mannwhitney_stat, mannwhitney_pval = mannwhitneyu(df_unn[cpg].values, df_gse[cpg].values)
    result['mw_stat'][cpg_id] = mannwhitney_stat
    result['mw_pval'][cpg_id] = mannwhitney_pval

    formula = f"Age + C(Dataset) + C(Sex) + Age"
    term = f"C(Dataset)[T.GSEUNN]"
    reg = smf.ols(formula=f"{cpg} ~ {formula}", data=df).fit()
    pvalues = dict(reg.pvalues)
    result[f"dataset_pval"][cpg_id] = pvalues[term]

result = correct_pvalues(result, ['mw_pval', 'dataset_pval'])
result = pd.DataFrame(result)
result.set_index("CpG", inplace=True)
result.sort_values([f"mw_pval"], ascending=[True], inplace=True)
result.to_excel(f"{path_save}/CpGs.xlsx", index=True)
result.to_pickle(f"{path_save}/CpGs.pkl")

# for thld_mv in thlds_mv:
#     for thld_var in thlds_var:
#         tmp_df = result.loc[(result['mw_pval_fdr_bh'] < thld_mv) & (result['variance'] > thld_var), :]
#         tmp_genes = get_genes_list(tmp_df, 'Gene', ['non-genic'], ';+')
#         np.savetxt(f"{path_save}/genes_mw({thld_mv})_var({thld_var}).txt", tmp_genes, fmt="%s")
#
# to_plot = result.head(num_cpgs_to_plot)
# for cpg_id, (cpg, row) in enumerate(to_plot.iterrows()):
#     fig = go.Figure()
#     add_box_trace(fig, df_unn[cpg].values, 'UNN')
#     add_box_trace(fig, df_gse[cpg].values, gse_dataset)
#     add_layout(fig, '', "Methylation Level", f"{cpg} ({manifest.loc[cpg, 'Gene']}): {row['mw_pval_fdr_bh']:0.4e}")
#     fig.update_layout({'colorway': ['blue', "red"]})
#     save_figure(fig, f"{path_save}/figs/cpgs/{cpg_id}_{cpg}_box")
#     fig = go.Figure()
#     add_violin_trace(fig, df_unn[cpg].values, 'UNN')
#     add_violin_trace(fig, df_gse[cpg].values, gse_dataset)
#     add_layout(fig, '', "Methylation Level", f"{cpg} ({manifest.loc[cpg, 'Gene']}): {row['mw_pval_fdr_bh']:0.4e}")
#     fig.update_layout({'colorway': ['blue', 'red']})
#     save_figure(fig, f"{path_save}/figs/cpgs/{cpg_id}_{cpg}_vio")
