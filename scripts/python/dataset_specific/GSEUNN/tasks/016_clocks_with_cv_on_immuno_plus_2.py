import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import random
import copy
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scripts.python.pheno.datasets.filter import filter_pheno
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from scripts.python.routines.plot.scatter import add_scatter_trace
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
import pathlib
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout, get_axis
from scripts.python.routines.plot.p_value import add_p_value_annotation


def calc_metrics(model, X, y, comment, params):
    y_pred = model.predict(X)
    df = pd.DataFrame({'y_pred': y_pred, 'y': y})
    score_2 = r2_score(y, y_pred)
    score_3 = model.score(X, y)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    params[f'{comment} R2'] = score_3
    params[f'{comment} RMSE'] = rmse
    params[f'{comment} MAE'] = mae
    return y_pred


dataset = "GSEUNN"
path = f"E:/YandexDisk/Work/pydnameth/datasets"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
manifest = get_manifest(platform)
status_col = get_column_name(dataset, 'Status').replace(' ','_')
age_col = get_column_name(dataset, 'Age').replace(' ','_')
sex_col = get_column_name(dataset, 'Sex').replace(' ','_')
status_dict = get_status_dict(dataset)
status_passed_fields = status_dict['Control'] + status_dict['Case']
sex_dict = get_sex_dict(dataset)
continuous_vars = {}
categorical_vars = {status_col: [x.column for x in status_passed_fields], sex_col: list(sex_dict.values())}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)

ctrl = pheno.loc[pheno['Group'] == 'Control']
esrd = pheno.loc[pheno['Group'] == 'ESRD']

clock_name = 'ipAGE_cv_plus_2'
path_save = f"{path}/{platform}/{dataset}/special/016_clocks_with_cv_on_immuno_plus_2/{clock_name}"
pathlib.Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    immuno_features = f.read().splitlines()

features = immuno_features + ['GDF15', 'FGF21']
target = 'Age'
scoring = 'r2'

k = 5
n_repeats = 5
random_state = 1
k_fold = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=1)

best_error = np.PINF
best_model = None
best_params = None
best_train_idx = None
best_val_idx = None
for train_idx, val_idx in k_fold.split(range(ctrl.shape[0])):
    X_train = ctrl.loc[ctrl.index[train_idx], features].to_numpy()
    y_train = ctrl.loc[ctrl.index[train_idx], target].to_numpy()
    X_val = ctrl.loc[ctrl.index[val_idx], features].to_numpy()
    y_val = ctrl.loc[ctrl.index[val_idx], target].to_numpy()

    # CV for detecting best params for ElasticNet
    cv = RepeatedKFold(n_splits=3, n_repeats=5, random_state=1)
    model_type = ElasticNet(max_iter=10000, tol=0.01)

    alphas = np.logspace(-5, np.log10(1.3 + 0.7 * random.uniform(0, 1)), 21)
    # alphas = np.logspace(-5, 1, 101)
    # alphas = [10]
    # l1_ratios = np.linspace(0.0, 1.0, 11)
    l1_ratios = [0.5]

    grid = dict()
    grid['alpha'] = alphas
    grid['l1_ratio'] = l1_ratios

    search = GridSearchCV(estimator=model_type, scoring=scoring, param_grid=grid, cv=cv, verbose=0)
    results = search.fit(X_train, y_train)

    model = results.best_estimator_
    score = model.score(X_train, y_train)

    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)

    if best_error > rmse:
        best_model = model
        best_error = rmse
        best_params = results.best_params_
        best_train_idx = train_idx
        best_val_idx = val_idx

print(f"Best RMSE in test: {best_error}")

params = copy.deepcopy(best_params)
model_dict = {'feature': ['Intercept'], 'coef': [best_model.intercept_]}
num_features = 0
for f_id, f in enumerate(features):
    coef = best_model.coef_[f_id]
    if abs(coef) > 0:
        model_dict['feature'].append(f)
        if f == 'Sex_ord_enc':
            print("Sex included!")
        model_dict['coef'].append(coef)
        num_features += 1
model_df = pd.DataFrame(model_dict)
model_df.to_excel(f'{path_save}/clock.xlsx', index=False)
with open(f'{path_save}/clock.pkl', 'wb') as handle:
    pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

y_pred_ctrl_train = calc_metrics(best_model, ctrl.loc[ctrl.index[best_train_idx], features].to_numpy(), ctrl.loc[ctrl.index[best_train_idx], target].to_numpy(), 'Control_train', params)
y_pred_ctrl_val = calc_metrics(best_model, ctrl.loc[ctrl.index[best_val_idx], features].to_numpy(), ctrl.loc[ctrl.index[best_val_idx], target].to_numpy(), 'Control_val', params)
y_pred_esrd_val = calc_metrics(best_model, esrd[features].to_numpy(), esrd[target].to_numpy(), 'ESRD', params)
y_pred_all = calc_metrics(best_model, pheno[features].to_numpy(), pheno[target].to_numpy(), 'All', params)
params['num_features'] = num_features
params_df = pd.DataFrame({'Feature': list(params.keys()), 'Value': list(params.values())})
params_df.to_excel(f'{path_save}/params.xlsx', index=False)
print(params_df)
pheno[clock_name] = y_pred_all
ctrl = pheno.loc[pheno['Group'] == 'Control']
ctrl_train = ctrl.loc[ctrl.index[best_train_idx], :]
ctrl_val = ctrl.loc[ctrl.index[best_val_idx], :]
esrd = pheno.loc[pheno['Group'] == 'ESRD']

ctrl_train_color = 'lime'
ctrl_val_color = 'cyan'
esrd_color = 'fuchsia'
dist_num_bins = 25

formula = f"{clock_name} ~ Age"
model_linear = smf.ols(formula=formula, data=ctrl_train).fit()
ctrl_train[f"{clock_name}_acceleration"] = ctrl_train[f'{clock_name}'] - model_linear.predict(ctrl_train)
ctrl_val[f"{clock_name}_acceleration"] = ctrl_val[f'{clock_name}'] - model_linear.predict(ctrl_val)
esrd[f"{clock_name}_acceleration"] = esrd[f'{clock_name}'] - model_linear.predict(esrd)
pheno[f"{clock_name}_acceleration"] = pheno[f'{clock_name}'] - model_linear.predict(pheno)
pheno.to_excel(f'{path_save}/pheno.xlsx', index=True)

values_ctrl_train = ctrl_train.loc[:, f"{clock_name}_acceleration"].values
values_ctrl_val = ctrl_val.loc[:, f"{clock_name}_acceleration"].values
values_esrd = esrd.loc[:, f"{clock_name}_acceleration"].values

stat_01, pval_01 = mannwhitneyu(values_ctrl_train, values_ctrl_val, alternative='two-sided')
stat_02, pval_02 = mannwhitneyu(values_ctrl_train, values_esrd, alternative='two-sided')
stat_12, pval_12 = mannwhitneyu(values_ctrl_val, values_esrd, alternative='two-sided')

fig = go.Figure()
fig.add_trace(
    go.Violin(
        y=values_ctrl_train,
        name=f"Control (train)",
        box_visible=True,
        meanline_visible=True,
        showlegend=True,
        line_color='black',
        fillcolor=ctrl_train_color,
        marker=dict(color=ctrl_train_color, line=dict(color='black', width=0.3), opacity=0.8),
        points='all',
        bandwidth=np.ptp(values_ctrl_train) / dist_num_bins,
        opacity=0.8
    )
)
fig.add_trace(
    go.Violin(
        y=values_ctrl_val,
        name=f"Control (val)",
        box_visible=True,
        meanline_visible=True,
        showlegend=True,
        line_color='black',
        fillcolor=ctrl_val_color,
        marker=dict(color=ctrl_val_color, line=dict(color='black', width=0.3), opacity=0.8),
        points='all',
        bandwidth=np.ptp(values_ctrl_val) / dist_num_bins,
        opacity=0.8
    )
)
fig.add_trace(
    go.Violin(
        y=values_esrd,
        name=f"ESRD",
        box_visible=True,
        meanline_visible=True,
        showlegend=True,
        line_color='black',
        fillcolor=esrd_color,
        marker=dict(color=esrd_color, line=dict(color='black', width=0.3), opacity=0.8),
        points='all',
        bandwidth=np.ptp(values_esrd) / 50,
        opacity=0.8
    )
)
add_layout(fig, "", "ipAGE (with GDF15, FGF22)<br>acceleration", f"")
fig = add_p_value_annotation(fig, {(0,1): pval_01, (1, 2) : pval_12, (0,2): pval_02})
fig.update_layout(title_xref='paper')
fig.update_layout(legend_font_size=20)
fig.update_layout(
    margin=go.layout.Margin(
        l=140,
        r=20,
        b=50,
        t=80,
        pad=0
    )
)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.25,
        xanchor="center",
        x=0.5
    )
)
save_figure(fig, f"{path_save}/venn")


for metric, name in {f"{clock_name}": "Estimated age", f"{clock_name}_acceleration": "Estimated age acceleration"}.items():
    stat, pval = mannwhitneyu(pheno.loc[pheno['Disease outcome'] == 'alive', metric].values, pheno.loc[pheno['Disease outcome'] == 'dead', metric].values, alternative='two-sided')
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=pheno.loc[pheno['Disease outcome'] == 'alive', metric].values,
            name=f"Alive",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor='blue',
            marker=dict(color='blue', line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(pheno.loc[pheno['Disease outcome'] == 'alive', metric].values) / dist_num_bins,
            opacity=0.8
        )
    )
    fig.add_trace(
        go.Violin(
            y=pheno.loc[pheno['Disease outcome'] == 'dead', metric].values,
            name=f"Dead",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor='red',
            marker=dict(color='red', line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(values_esrd) / 50,
            opacity=0.8
        )
    )
    add_layout(fig, "", name, f"Mann-Whitney p-value: {pval:0.2e}")
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_layout(
        margin=go.layout.Margin(
            l=140,
            r=20,
            b=50,
            t=80,
            pad=0
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5
        )
    )
    save_figure(fig, f"{path_save}/{metric}")


fig = go.Figure()
add_scatter_trace(fig, ctrl_train.loc[:, target].values, ctrl_train.loc[:, f"{clock_name}"].values, f"Control (train)")
add_scatter_trace(fig, ctrl_train.loc[:, target].values, model_linear.fittedvalues.values, "", "lines")
add_scatter_trace(fig, ctrl_val.loc[:, target].values, ctrl_val.loc[:, f"{clock_name}"].values, f"Control (val)")
add_scatter_trace(fig, esrd.loc[:, target].values, esrd.loc[:, f"{clock_name}"].values, f"ESRD")
add_layout(fig, f"{target}", 'ipAGE (with GDF15, FGF22)', f"")
fig.update_layout({'colorway': [ctrl_train_color, ctrl_train_color, ctrl_val_color, esrd_color]})
fig.update_layout(legend_font_size=20)
fig.update_layout(
    margin=go.layout.Margin(
        l=120,
        r=20,
        b=75,
        t=45,
        pad=0
    )
)
fig.update_yaxes(autorange=False)
fig.update_xaxes(autorange=False)
fig.update_layout(yaxis_range=[0, 150])
fig.update_layout(xaxis_range=[10, 100])
save_figure(fig, f"{path_save}/scatter")

