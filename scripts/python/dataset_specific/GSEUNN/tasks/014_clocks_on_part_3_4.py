import pandas as pd
import numpy as np
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
ord_enc = OrdinalEncoder()
pheno["Sex_ord_enc"] = ord_enc.fit_transform(pheno[["Sex"]])

ctrl = pheno.loc[pheno['Group'] == 'Control']
esrd = pheno.loc[pheno['Group'] == 'ESRD']

thld_abs_diff = 10000
part_3_4 = pd.read_excel(f"{path}/{platform}/{dataset}/data/immuno/part3_part4_with_age_sex.xlsx", index_col='ID')
ord_enc = OrdinalEncoder()
part_3_4["Sex_ord_enc"] = ord_enc.fit_transform(part_3_4[["Sex"]])
part_3_4[f'ImmunoAgeDiff'] = part_3_4[f'ImmunoAge'] - part_3_4[f'Age']
part_3_4 = part_3_4[~part_3_4.index.str.startswith(('Q', 'H'))]
part_3_4 = part_3_4[abs(part_3_4[f'ImmunoAgeDiff']) <= thld_abs_diff]
rmse_part_3_4 = np.sqrt(mean_squared_error(part_3_4.loc[:, 'Age'].values, part_3_4.loc[:, 'ImmunoAge'].values))
mae_part_3_4 = mean_absolute_error(part_3_4.loc[:, 'Age'].values, part_3_4.loc[:, 'ImmunoAge'].values)
print(f"RMSE in part_3_4: {rmse_part_3_4}")
print(f"MAE in part_3_4: {mae_part_3_4}")

clock_name = 'ipAGE_part_3_4'

path_save = f"{path}/{platform}/{dataset}/special/014_clocks_on_part_3_4/{clock_name}_{thld_abs_diff}"
pathlib.Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    immuno_features = f.read().splitlines()

features = immuno_features
target = 'Age'
scoring = 'r2'

X_train = part_3_4[features].to_numpy()
y_train = part_3_4[target].to_numpy()

cv = RepeatedKFold(n_splits=3, n_repeats=10, random_state=1)
model_type = ElasticNet(max_iter=10000, tol=0.01)

alphas = np.logspace(-5, np.log10(2.3 + 0.7 * random.uniform(0, 1)), 51)
alphas = np.logspace(-5, 1, 101)
# l1_ratios = np.linspace(0.0, 1.0, 11)
l1_ratios = [0.5]

grid = dict()
grid['alpha'] = alphas
grid['l1_ratio'] = l1_ratios

search = GridSearchCV(estimator=model_type, scoring=scoring, param_grid=grid, cv=cv, verbose=3)
results = search.fit(X_train, y_train)

model = results.best_estimator_

score = model.score(X_train, y_train)
params = copy.deepcopy(results.best_params_)

searching_process = pd.DataFrame(results.cv_results_)
searching_process.to_excel(f'{path_save}/searching_process_{scoring}.xlsx', index=False)

model_dict = {'feature': ['Intercept'], 'coef': [model.intercept_]}
num_features = 0
for f_id, f in enumerate(features):
    coef = model.coef_[f_id]
    if abs(coef) > 0:
        model_dict['feature'].append(f)
        if f == 'Sex_ord_enc':
            print("Sex included!")
        model_dict['coef'].append(coef)
        num_features += 1
model_df = pd.DataFrame(model_dict)
model_df.to_excel(f'{path_save}/clock.xlsx', index=False)
with open(f'{path_save}/clock.pkl', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

y_pred_ctrl = calc_metrics(model, ctrl[features].to_numpy(), ctrl[target].to_numpy(), 'Control', params)
y_pred_esrd = calc_metrics(model, esrd[features].to_numpy(), esrd[target].to_numpy(), 'ESRD', params)
y_pred_all = calc_metrics(model, pheno[features].to_numpy(), pheno[target].to_numpy(), 'All', params)
y_pred_part_3_4 = calc_metrics(model, part_3_4[features].to_numpy(), part_3_4[target].to_numpy(), 'part_3_4', params)
params['num_features'] = num_features
params_df = pd.DataFrame({'Feature': list(params.keys()), 'Value': list(params.values())})
params_df.to_excel(f'{path_save}/params.xlsx', index=False)

print(params_df)
pheno[clock_name] = y_pred_all
part_3_4[clock_name] = y_pred_part_3_4
ctrl = pheno.loc[pheno['Group'] == 'Control']
esrd = pheno.loc[pheno['Group'] == 'ESRD']
pheno.to_excel(f'{path_save}/pheno.xlsx', index=True)

ctrl_color = 'lime'
part_3_4_color = 'cyan'
esrd_color = 'fuchsia'
dist_num_bins = 25

formula = f"{clock_name} ~ Age"
model_linear = smf.ols(formula=formula, data=ctrl).fit()
ctrl[f"{clock_name}_acceleration"] = ctrl[f'{clock_name}'] - model_linear.predict(ctrl)
esrd[f"{clock_name}_acceleration"] = esrd[f'{clock_name}'] - model_linear.predict(esrd)
part_3_4[f"{clock_name}_acceleration"] = part_3_4[f'{clock_name}'] - model_linear.predict(part_3_4)

values_ctrl = ctrl.loc[:, f"{clock_name}_acceleration"].values
values_part_3_4 = part_3_4.loc[:, f"{clock_name}_acceleration"].values
values_esrd = esrd.loc[:, f"{clock_name}_acceleration"].values

stat_01, pval_01 = mannwhitneyu(values_ctrl, values_part_3_4, alternative='two-sided')
stat_02, pval_02 = mannwhitneyu(values_ctrl, values_esrd, alternative='two-sided')
stat_12, pval_12 = mannwhitneyu(values_part_3_4, values_esrd, alternative='two-sided')

fig = go.Figure()
fig.add_trace(
    go.Violin(
        y=values_ctrl,
        name=f"Control",
        box_visible=True,
        meanline_visible=True,
        showlegend=True,
        line_color='black',
        fillcolor=ctrl_color,
        marker=dict(color=ctrl_color, line=dict(color='black', width=0.3), opacity=0.8),
        points='all',
        bandwidth=np.ptp(values_ctrl) / dist_num_bins,
        opacity=0.8
    )
)
fig.add_trace(
    go.Violin(
        y=values_part_3_4,
        name=f"Control (test)",
        box_visible=True,
        meanline_visible=True,
        showlegend=True,
        line_color='black',
        fillcolor=part_3_4_color,
        marker=dict(color=part_3_4_color, line=dict(color='black', width=0.3), opacity=0.8),
        points='all',
        bandwidth=np.ptp(values_part_3_4) / dist_num_bins,
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
add_layout(fig, "", "ipAGE (new) acceleration", f"")
fig.update_layout({'colorway': ['lime', 'cyan', 'fuchsia']})
fig = add_p_value_annotation(fig, {(0,1): pval_01, (1, 2) : pval_12, (0,2): pval_02})
fig.update_yaxes(autorange=False)
fig.update_layout(yaxis_range=[-300, 300])
fig.update_layout(title_xref='paper')
fig.update_layout(legend_font_size=20)
fig.update_layout(
    margin=go.layout.Margin(
        l=110,
        r=20,
        b=50,
        t=90,
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
fig.add_annotation(dict(font=dict(color='black', size=45),
                        x=-0.18,
                        y=1.4,
                        showarrow=False,
                        text=f"(b)",
                        textangle=0,
                        yanchor='top',
                        xanchor='left',
                        xref="paper",
                        yref="paper"))
save_figure(fig, f"{path_save}/venn")


fig = go.Figure()
add_scatter_trace(fig, ctrl.loc[:, target].values, ctrl.loc[:, f"{clock_name}"].values, f"Control")
add_scatter_trace(fig, ctrl.loc[:, target].values, model_linear.fittedvalues.values, "", "lines")
add_scatter_trace(fig, part_3_4.loc[:, target].values, part_3_4.loc[:, f"{clock_name}"].values, f"Control (test)")
add_scatter_trace(fig, esrd.loc[:, target].values, esrd.loc[:, f"{clock_name}"].values, f"ESRD")
add_layout(fig, f"{target}", 'ipAGE (new)', f"")
fig.update_layout({'colorway': ['lime', 'lime', 'cyan', 'fuchsia']})
fig.update_layout(legend_font_size=20)
fig.update_layout(
    margin=go.layout.Margin(
        l=80,
        r=20,
        b=80,
        t=65,
        pad=0
    )
)
fig.update_yaxes(autorange=False)
fig.update_xaxes(autorange=False)
fig.update_layout(yaxis_range=[10, 110])
fig.update_layout(xaxis_range=[10, 100])
fig.add_annotation(dict(font=dict(color='black', size=45),
                        x=-0.13,
                        y=1.20,
                        showarrow=False,
                        text=f"(a)",
                        textangle=0,
                        yanchor='top',
                        xanchor='left',
                        xref="paper",
                        yref="paper"))
save_figure(fig, f"{path_save}/scatter")
