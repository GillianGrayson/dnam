import pandas as pd
import plotly.graph_objects as go
from scripts.python.preprocessing.serialization.routines.filter import get_forbidden_cpgs, manifest_filter
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.preprocessing.serialization.routines.save import save_pheno_betas_to_pkl
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
from pathlib import Path
import numpy as np
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_default_statuses_ids, get_status_dict, get_default_statuses, get_sex_dict
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
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
from pathlib import Path
from scripts.python.routines.manifest import get_genes_list
from sklearn.feature_selection import VarianceThreshold
from scripts.python.routines.plot.histogram import add_histogram_trace
import statsmodels.formula.api as smf


path = f"E:/YandexDisk/Work/pydnameth/datasets/meta/tasks/unn_dataset_specific"
platform = 'GPL21145'

manifest = get_manifest(platform)

path_save = f"{path}/006_age_acceleration"
Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

calc_features = ["DNAmAge", "DNAmAgeHannum", "DNAmPhenoAge", "DNAmGrimAge"]
features = ["AgeAccelGrim", "DNAmAge", "CD8T", "CD4T", "NK", "Bcell", "Mono", "Gran", "propNeuron", "DNAmAgeHannum", "DNAmPhenoAge", "DNAmGDF15", "DNAmGrimAge", "IEAA", "EEAA", "IEAA.Hannum"]

pheno = pd.read_pickle(f"{path}/004_prepare_python_data/pheno.pkl")
pheno = pheno.drop(features, axis=1, errors='ignore')

calcs = pd.read_csv(f"{path}/005_prepare_for_calculator/betas.output.csv", delimiter=",", index_col='subject_id')
calcs = calcs[features]

df = pd.merge(pheno, calcs, left_index=True, right_index=True)
df.to_excel(f"{path}/006_age_acceleration/pheno_xtd.xlsx")
df.to_pickle(f"{path}/006_age_acceleration/pheno_xtd.pkl")

pheno_eu = df.loc[df['Sample_Group'] == 'EU', :]
pheno_ru = df.loc[df['Sample_Group'] == 'RU', :]

stat, pval = mannwhitneyu(pheno_eu[f"AgeAccelGrim"].values, pheno_ru[f"AgeAccelGrim"].values)
vio = go.Figure()
add_violin_trace(vio, pheno_eu[f"AgeAccelGrim"].values, 'EU')
add_violin_trace(vio, pheno_ru[f"AgeAccelGrim"].values, "RU")
add_layout(vio, "", f"AgeAccelGrim", f"Mann-Whitney p-value: {pval:0.4e}")
vio.update_layout({'colorway': ['blue', 'red']})
save_figure(vio, f"{path_save}/figs/Acc_vio_AgeAccelGrim")

metrics = ['R2', 'R2_adj', 'mw_stat_acc', 'mw_pval_acc', 'mw_stat_diff', 'mw_pval_diff']
res_dict = {"metric": calc_features}
for m in metrics:
    res_dict[m] = np.zeros(len(calc_features))
for f_id, f in enumerate(calc_features):
    formula = f"{f} ~ Age"
    reg = smf.ols(formula=formula, data=pheno_eu).fit()
    res_dict['R2'][f_id] = reg.rsquared
    res_dict['R2_adj'][f_id] = reg.rsquared_adj

    df[f"{f}Acc"] = df[f] - reg.predict(df)
    df[f"{f}Diff"] = df[f] - df["Age"]
    pheno_eu = df.loc[df['Sample_Group'] == 'EU', :]
    pheno_ru = df.loc[df['Sample_Group'] == 'RU', :]

    scatter = go.Figure()
    add_scatter_trace(scatter, [0, 100], [0, 100], "", "lines")
    add_scatter_trace(scatter, pheno_eu["Age"].values, pheno_eu[f].values, 'EU')
    add_scatter_trace(scatter, pheno_eu["Age"].values, reg.fittedvalues.values, "", "lines")
    add_scatter_trace(scatter, pheno_ru["Age"].values, pheno_ru[f].values, "RU")
    add_layout(scatter, "Age", f, "")
    scatter.update_layout({'colorway': ['black', 'blue', 'blue', 'red']})
    scatter.update_layout(
        margin=go.layout.Margin(
            l=80,
            r=20,
            b=80,
            t=50,
            pad=0
        )
    )
    scatter.update_yaxes(autorange=False)
    scatter.update_xaxes(autorange=False)
    scatter.update_layout(yaxis_range=[10, 100])
    scatter.update_layout(xaxis_range=[10, 100])
    save_figure(scatter, f"{path_save}/figs/scatter_Age_{f}")

    stat, pval = mannwhitneyu(pheno_eu[f"{f}Acc"].values, pheno_ru[f"{f}Acc"].values)
    res_dict['mw_stat_acc'][f_id] = stat
    res_dict['mw_pval_acc'][f_id] = pval
    vio = go.Figure()
    add_violin_trace(vio, pheno_eu[f"{f}Acc"].values, 'EU')
    add_violin_trace(vio, pheno_ru[f"{f}Acc"].values, "RU")
    add_layout(vio, "", f"{f} acceleration", f"Mann-Whitney p-value: {pval:0.4e}")
    vio.update_layout({'colorway': ['blue', 'red']})
    save_figure(vio, f"{path_save}/figs/Acc_vio_{f}")

    stat, pval = mannwhitneyu(pheno_eu[f"{f}Diff"].values, pheno_ru[f"{f}Diff"].values)
    res_dict['mw_stat_diff'][f_id] = stat
    res_dict['mw_pval_diff'][f_id] = pval
    vio = go.Figure()
    add_violin_trace(vio, pheno_eu[f"{f}Diff"].values, "EU")
    add_violin_trace(vio, pheno_ru[f"{f}Diff"].values, "RU")
    add_layout(vio, "", f"{f} - Age", f"Mann-Whitney p-value: {pval:0.4e}")
    vio.update_layout({'colorway': ['blue', 'red']})
    save_figure(vio, f"{path_save}/figs/Diff_{f}")
