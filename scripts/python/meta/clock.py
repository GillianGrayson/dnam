import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout
import os
import pickle
from scripts.python.pheno.datasets.filter import filter_pheno
from scripts.python.pheno.datasets.features import get_column_name, get_status_names_dict, get_status_dict, \
    get_sex_dict

path = f"E:/YandexDisk/Work/pydnameth/datasets"

datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')

datasets_train = ["GSE84727", "GSE147221", "GSE125105", "GSE111629", "GSE128235", "GSE72774"]
datasets_test = ["GSEUNN", "GSE147221", "GSE84727", "GSE125105", "GSE111629", "GSE128235", "GSE72774", "GSE53740", "GSE144858"]

dnam_acc_type = 'DNAmGrimAgeAcc'

target = f"correct_Age(0.05)_Status(0.05)"

path_save = f"{path}/meta/EWAS/{target}"
if not os.path.exists(f"{path_save}/clock"):
    os.makedirs(f"{path_save}/clock")

pheno_all = pd.DataFrame(columns=['Age', 'Status'])
pheno_all.index.name = 'subject_id'
for d_id, dataset in enumerate(datasets_train):

    platform = datasets_info.loc[dataset, 'platform']
    manifest = get_manifest(platform)

    print(dataset)
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

    pheno = df[[age_col, status_col]]
    status_dict_inverse = dict((v, k) for k, v in status_dict.items())
    pheno.replace({status_col:status_dict_inverse}, inplace=True)
    pheno.rename(columns={age_col: 'Age', status_col: 'Status'}, inplace=True)
    pheno_all = pheno_all.append(pheno, verify_integrity=True)

    cpgs = betas.columns.values
    betas = df[cpgs].T
    if d_id == 0:
        betas_all = betas
    else:
        betas_all = betas_all.merge(betas, how='inner', left_index=True, right_index=True)

betas_all = betas_all.T
betas_all.index.name = "subject_id"
df_all = pd.merge(pheno_all, betas_all, left_index=True, right_index=True)

with open(f"cpgs.txt") as f:
    cpgs_target = f.read().splitlines()
cpgs_target = list(set.intersection(set(betas_all.columns.values), set(cpgs_target)))

X_target = df_all.loc[df_all['Status'] == 'Control', cpgs_target].to_numpy()
y_target = df_all.loc[df_all['Status'] == 'Control', 'Age'].to_numpy()

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1337)

clock = ElasticNetCV(n_alphas=20, cv=cv, n_jobs=2, verbose=1)
clock.fit(X_target, y_target)

clock_dict = {'feature': ['Intercept'], 'coef': [clock.intercept_], 'default': [0.0]}
num_features = 0
for cpg_id, cpg in enumerate(cpgs_target):
    coef = clock.coef_[cpg_id]
    if abs(coef) > 0:
        clock_dict['feature'].append(cpg)
        clock_dict['coef'].append(coef)
        cpg_values = df_all.loc[df_all['Status'] == 'Control', cpg].to_numpy()
        mean_value = np.mean(cpg_values)
        clock_dict['default'].append(mean_value)
        num_features += 1
clock_df = pd.DataFrame(clock_dict)
clock_df.set_index('feature', inplace=True)
if not os.path.exists(f"{path_save}/clock/{num_features}"):
    os.makedirs(f"{path_save}/clock/{num_features}")
clock_df.to_excel(f"{path_save}/clock/{num_features}/clock.xlsx", index=True)
pickle.dump(clock, open(f"{path_save}/clock/{num_features}/clock.sav", 'wb'))
np.savetxt(f"{path_save}/clock/{num_features}/clock_cpgs.txt", cpgs_target, fmt='%s')

metrics_dict = {'alpha': clock.alpha_, 'l1_ratio': clock.l1_ratio_, 'num_features': num_features}
y_target_pred = clock.predict(X_target)
metrics_dict['R2_Control'] = clock.score(X_target, y_target)
metrics_dict['RMSE_Control'] = np.sqrt(mean_squared_error(y_target, y_target_pred))
metrics_dict['MAE_Control'] = mean_absolute_error(y_target, y_target_pred)
X_all = df_all.loc[:, cpgs_target].to_numpy()
y_all = df_all.loc[:, 'Age'].to_numpy()
y_all_pred = clock.predict(X_all)
metrics_dict['R2_All'] = clock.score(X_all, y_all)
metrics_dict['RMSE_All'] = np.sqrt(mean_squared_error(y_all, y_all_pred))
metrics_dict['MAE_All'] = mean_absolute_error(y_all, y_all_pred)
metrics_df = pd.DataFrame(metrics_dict, index=[0])
metrics_df.to_excel(f"{path_save}/clock/{num_features}/metrics.xlsx", index=False)

pheno_all[f'AgeEST'] = y_all_pred

formula = f"AgeEST ~ Age"
reg = smf.ols(formula=formula, data=pheno_all.loc[pheno_all['Status'] == 'Control', :]).fit()
res_dict = {'R2': reg.rsquared, 'R2_adj': reg.rsquared_adj}
res_dict['RMSE'] = np.sqrt(mean_squared_error(pheno_all.loc[pheno_all['Status'] == 'Control', 'Age'].values), reg.fittedvalues.values)
res_dict['MAE'] = mean_absolute_error(pheno_all.loc[pheno_all['Status'] == 'Control', 'Age'].values, reg.fittedvalues.values)

pheno_all['Acceleration'] = pheno_all[f'AgeEST'] - reg.predict(pheno_all)
pheno_all.to_excel(f"{path_save}/clock/{num_features}/pheno.xlsx", index=True)

scatter = go.Figure()
add_scatter_trace(scatter, pheno_all.loc[pheno_all['Status'] == 'Case', 'Age'].values, pheno_all.loc[pheno_all['Status'] == 'Case', 'AgeEST'].values, 'Case')
add_scatter_trace(scatter, pheno_all.loc[pheno_all['Status'] == 'Control', 'Age'].values, pheno_all.loc[pheno_all['Status'] == 'Control', 'AgeEST'].values, 'Control')
add_scatter_trace(scatter, pheno_all.loc[pheno_all['Status'] == 'Control', 'Age'].values, reg.fittedvalues.values, "", "lines")
add_layout(scatter, "Age", "AgeEST", f"Control: R2: {res_dict['R2']:0.2f}, RMSE: {res_dict['RMSE']:0.2f}, MAE: {res_dict['MAE']:0.2f}")
scatter.update_layout({'colorway': ['red', 'blue', 'blue']})
save_figure(scatter, f"{path_save}/clock/{num_features}/scatter_Age_AgeEST")

statistic, pvalue = mannwhitneyu(pheno_all.loc[pheno_all['Status'] == 'Control', 'Acceleration'].values, pheno_all.loc[pheno_all['Status'] == 'Case', 'Acceleration'].values)
res_dict['MW_statistic'] = statistic
res_dict['MW_pvalue'] = pvalue
res_df = pd.DataFrame(res_dict, index=[0])
res_df.to_excel(f"{path_save}/clock/{num_features}/res.xlsx", index=False)

box = go.Figure()
add_violin_trace(box, pheno_all.loc[pheno_all['Status'] == 'Control', 'Acceleration'].values, 'Control')
add_violin_trace(box, pheno_all.loc[pheno_all['Status'] == 'Case', 'Acceleration'].values, 'Case')
add_layout(box, "", 'Acceleration', f"p-val = {pvalue:0.4e}")
box.update_layout({'colorway': ['blue', 'red']})
save_figure(box, f"{path_save}/clock/{num_features}/box_Acceleration")

for d_id, dataset in enumerate(datasets_test):
    print(dataset)

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

    df_1 = df.loc[(df[status_col] == status_dict['Control']), :]
    formula = f"AgeEST ~ {age_col}"
    reg = smf.ols(formula=formula, data=df_1).fit()
    R2 = reg.rsquared
    R2_adj = reg.rsquared_adj
    RMSE = np.sqrt(mean_squared_error(df_1[age_col].values, reg.fittedvalues.values))
    MAE = mean_absolute_error(df_1[age_col].values, reg.fittedvalues.values)

    df['AgeESTAcc'] = df[f'AgeEST']  - reg.predict(pheno)
    df_1 = df.loc[(df[status_col] == status_dict['Control']), :]
    df_2 = df.loc[(df[status_col] == status_dict['Case']), :]

    if not os.path.exists(f"{path_save}/clock/{num_features}/{dataset}"):
        os.makedirs(f"{path_save}/clock/{num_features}/{dataset}")

    scatter = go.Figure()
    add_scatter_trace(scatter, df_1[age_col].values, df_1['AgeEST'].values, status_names_dict['Control'])
    add_scatter_trace(scatter, df_1[age_col].values, reg.fittedvalues.values, "", "lines")
    add_scatter_trace(scatter, df_2[age_col].values, df_2['AgeEST'].values, status_names_dict['Case'])
    add_layout(scatter, "Age", "AgeEST", f"Control: R2: {R2:0.2f}, RMSE: {RMSE:0.2f}, MAE: {MAE:0.2f}")
    scatter.update_layout({'colorway': ['blue', 'blue', 'red']})
    save_figure(scatter, f"{path_save}/clock/{num_features}/{dataset}/scatter_Age_AgeEST")

    statistic, pvalue = mannwhitneyu(df_1['AgeESTAcc'].values, df_2['AgeESTAcc'].values)

    vio = go.Figure()
    add_violin_trace(vio, df_1['AgeESTAcc'].values, status_names_dict['Control'])
    add_violin_trace(vio, df_2['AgeESTAcc'].values, status_names_dict['Case'])
    add_layout(vio, "", 'AgeESTAcc', f"p-value: {pvalue:0.4e}")
    vio.update_layout({'colorway': ['blue', 'red']})
    save_figure(vio, f"{path_save}/clock/{num_features}/{dataset}/vio_AgeESTAcc")
