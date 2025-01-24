import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
from scipy.stats import kruskal
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout, get_axis
import os
import pickle
from scripts.python.pheno.datasets.filter import filter_pheno
from scripts.python.pheno.datasets.features import get_column_name, get_status_names_dict, get_status_dict, \
    get_sex_dict

path = f"E:/YandexDisk/Work/pydnameth/datasets"

datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

target = f"Age_Status"
path_load = f"{path}/meta/EWAS/{target}"
path_save = f"{path}/meta/EWAS/strange_tasks/meta_ctrls_age_acc"
if not os.path.exists(f"{path_save}"):
    os.makedirs(f"{path_save}")
num_features = 29

datasets = ["GSE84727", "GSE147221", "GSE125105", "GSE111629", "GSE128235", "GSE72774"]

clock = pickle.load(open(f"{path_load}/clock/{num_features}/clock.sav", 'rb'))
clock_df = pd.read_excel(f"{path_load}/clock/{num_features}/clock.xlsx", index_col='feature')
cpgs_target = np.loadtxt(f"{path_load}/clock/{num_features}/clock_cpgs.txt", dtype='str')

controls_age_acc = {}

for d_id, dataset in enumerate(datasets):

    platform = datasets_info.loc[dataset, 'platform']
    manifest = get_manifest(platform)

    status_col = get_column_name(dataset, 'Status').replace(' ', '_')
    age_col = get_column_name(dataset, 'Age').replace(' ', '_')
    sex_col = get_column_name(dataset, 'Sex').replace(' ', '_')
    status_dict = get_status_dict(dataset)
    status_vals = sorted(list(status_dict.values()))
    status_names_dict = get_status_names_dict(dataset)
    sex_dict = get_sex_dict(dataset)

    continuous_vars = {'Age': age_col}
    categorical_vars = {status_col: status_dict, sex_col: sex_dict}
    pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
    pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
    betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
    na_cols = betas.columns[betas.isna().any()].tolist()
    if len(na_cols) > 0:
        print(f"CpGs with NaNs in {dataset}: {na_cols}")
        s = betas.stack(dropna=False)
        na_pairs = [list(x) for x in s.index[s.isna()]]
        print(*na_pairs, sep='\n')
    betas.dropna(axis='columns', how='any', inplace=True)
    df = pd.merge(pheno, betas, left_index=True, right_index=True)

    df_for_clock = df.reindex(columns=cpgs_target, fill_value=0.0)
    missing_cpgs = list(set(cpgs_target) - set(df.columns))
    for cpg in missing_cpgs:
        if cpg in clock_df.index:
            df_for_clock[cpg] = clock_df.loc[cpg, 'default']
        else:
            df_for_clock[cpg] = 0

    df[f'AgeEST'] = clock.predict(df_for_clock.to_numpy())

    df_controls = df.loc[(df[status_col] == status_dict['Control']), :]
    formula = f"AgeEST ~ {age_col}"
    reg = smf.ols(formula=formula, data=df_controls).fit()

    df_controls['AgeESTAcc'] = df_controls[f'AgeEST'] - reg.predict(df_controls)
    controls_age_acc[dataset] = df_controls.loc[:, 'AgeESTAcc'].values

stat, pval = kruskal(*controls_age_acc.values())
print('Statistics=%.3f, p=%.3f' % (stat, pval))

box = go.Figure()
for key, val in controls_age_acc.items():
    add_violin_trace(box, val, key)
add_layout(box, "Dataset", 'Age-acceleration for Controls', f"Kruskal-Wallis p-value = {pval:0.4e}")
box.update_layout({'colorway': ['blue']*len(controls_age_acc.keys())})
box.update_layout(
    template="none",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    title=dict(
        text=f"Kruskal-Wallis p-value = {pval:0.4e}",
        font=dict(
            size=20
        )
    ),
    autosize=True,
    margin=go.layout.Margin(
        l=120,
        r=20,
        b=80,
        t=100,
        pad=0
    ),
    showlegend=False,
    xaxis=get_axis("Dataset", 20, 10),
    yaxis=get_axis('Age-acceleration for Controls', 20, 20),
)
save_figure(box, f"{path_save}/box")
