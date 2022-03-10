import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.python.routines.betas import betas_drop_na
import pickle
import random
import plotly.express as px
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
from statsmodels.stats.multitest import multipletests


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
betas = pd.read_pickle(f"{path}/{platform}/{dataset}/betas.pkl")
betas = betas_drop_na(betas)

df = pd.merge(pheno, betas, left_index=True, right_index=True)
df.set_index('ID', inplace=True)
df_ctrl = df.loc[(df[status_col] == 'Control'), :]
df_case = df.loc[(df[status_col] == 'ESRD'), :]

path_save = f"{path}/{platform}/{dataset}/special/020_agena"
pathlib.Path(f"{path_save}/figs/cpgs").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/figs/subjects").mkdir(parents=True, exist_ok=True)

agena = pd.read_excel(f"{path}/{platform}/{dataset}/data/agena/proc.xlsx", index_col='feature')
agena = agena.T
agena.index.name = "subject_id"
agena_cpgs = list(set(agena.columns.values) - set(['Group']))
agena.loc[:, agena_cpgs] *= 0.01

subjects_common = sorted(list(set(agena.index.values).intersection(set(df_ctrl.index.values))))
subjects_agena_only = set(agena.index.values) - set(df_ctrl.index.values)
cpgs_common = sorted(list(set(agena_cpgs).intersection(set(betas.columns.values))))

rel_diff_df = pd.DataFrame(index=subjects_common, columns=cpgs_common+['Group'])

for subject in subjects_common:
    agena_i = agena.loc[subject, agena_cpgs]
    agena_i.dropna(how='all')
    cpgs_i = sorted(list(set(agena_i.index.values).intersection(set(betas.columns.values))))
    df_i = df_ctrl.loc[subject, cpgs_i]

    rel_diff_df.at[subject, 'Group'] = agena.at[subject, 'Group']

    fig = go.Figure()
    for cpg_id, cpg in enumerate(cpgs_i):
        distrib_i = df_ctrl.loc[:, cpg].values
        fig.add_trace(
            go.Violin(
                x=[cpg] * len(distrib_i),
                y=distrib_i,
                box_visible=True,
                meanline_visible=True,
                line_color='grey',
                showlegend=False,
                opacity=1.0
            )
        )

        showlegend = False
        if cpg_id == 0:
            showlegend = True

        meth_epic = df_ctrl.at[subject, cpg]
        meth_agena = agena_i.at[cpg]
        tmp = (meth_agena - meth_epic) / meth_epic * 100.0
        rel_diff_df.at[subject, cpg] = tmp

        fig.add_trace(
            go.Scatter(
                x=[cpg],
                y=[meth_epic],
                showlegend=showlegend,
                name="850K",
                mode="markers",
                marker=dict(
                    size=15,
                    opacity=0.7,
                    line=dict(
                        width=1
                    ),
                    color='red'
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[cpg],
                y=[meth_agena],
                showlegend=showlegend,
                name="Agena",
                mode="markers",
                marker=dict(
                    size=12,
                    opacity=0.7,
                    line=dict(
                        width=1
                    ),
                    color='blue'
                ),
            )
        )

    add_layout(fig, "", 'Methylation level', f"")
    fig.update_xaxes(tickangle=270)
    fig.update_xaxes(tickfont_size=15)
    fig.update_layout(margin=go.layout.Margin(
        l=80,
        r=20,
        b=120,
        t=50,
        pad=0
    ))
    save_figure(fig, f"{path_save}/figs/subjects/{subject}")

colors = px.colors.qualitative.Set1
groups = sorted(rel_diff_df['Group'].unique())
rel_diff_df.to_excel(f"{path_save}/rel_diff.xlsx", index=True)

fig = go.Figure()
for cpg_id, cpg in enumerate(cpgs_common):
    series_i = rel_diff_df.loc[subjects_common, cpg].dropna()
    series_i = series_i.astype('float64')
    distrib_i = series_i.values

    showlegend = False
    if cpg_id == 0:
        showlegend = True

    fig.add_trace(
        go.Violin(
            x=[cpg] * len(distrib_i),
            y=distrib_i,
            showlegend=False,
            box_visible=True,
            meanline_visible=True,
            line_color='black',
            line=dict(width=0.35),
            fillcolor='grey',
            marker=dict(color='grey', line=dict(color='black', width=0.3), opacity=0.8),
            points=False,
            bandwidth=np.ptp(distrib_i) / 25,
            opacity=0.8
        )
    )
    for g_id, g in enumerate(groups):
        series_i = rel_diff_df.loc[rel_diff_df['Group'] == g, cpg].dropna()
        series_i = series_i.astype('float64')
        distrib_i = series_i.values
        fig.add_trace(
            go.Box(
                x=[cpg] * len(distrib_i),
                name=g,
                y=distrib_i,
                boxpoints='all',
                fillcolor='rgba(255,255,255,0)',
                hoveron = 'points',
                line = {'color': 'rgba(255,255,255,0)'},
                pointpos = -2,
                showlegend = showlegend,
                marker=dict(size=4, color=colors[g_id], line=dict(color='black', width=0.3), opacity=0.6),
            )
        )

add_layout(fig, "", "Relative difference, %", f"")
fig.update_xaxes(tickangle=270)
fig.update_xaxes(tickfont_size=15)
fig.update_layout(margin=go.layout.Margin(
    l=120,
    r=20,
    b=120,
    t=50,
    pad=0
))
fig.update_layout(title_xref='paper')
fig.update_layout(legend= {'itemsizing': 'constant'})
fig.update_layout(legend_font_size=20)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)
save_figure(fig, f"{path_save}/figs/rel_diff")

pvals = []
values_dict = {'ID': subjects_common}
for cpg_id, cpg in enumerate(cpgs_common):
    values_dict[f"{cpg}_850K"] = []
    values_dict[f"{cpg}_agena"] = []
    epic_data = []
    agena_data = []
    for subject in subjects_common:
        meth_epic = df_ctrl.at[subject, cpg]
        epic_data.append(meth_epic)
        meth_agena = agena.at[subject, cpg]
        agena_data.append(meth_agena)
        values_dict[f"{cpg}_850K"].append(meth_epic)
        values_dict[f"{cpg}_agena"].append(meth_agena)
    stat, pval = mannwhitneyu(epic_data, agena_data, alternative='two-sided')
    pvals.append(pval)

values_df = pd.DataFrame(values_dict)
values_df.set_index("ID", inplace=True)
values_df.to_excel(f"{path_save}/values.xlsx", index=True)

_, pvals_corr, _, _ = multipletests(pvals, 0.05, method='fdr_bh')

pvals_df = pd.DataFrame(index=cpgs_common)
pvals_df['pvals'] = pvals
pvals_df['pvals_fdr_bh'] = pvals_corr
pvals_df.to_excel(f"{path_save}/pvals.xlsx", index=True)

for cpg_id, cpg in enumerate(cpgs_common):

    epic_data = []
    agena_data = []
    for subject in subjects_common:
        meth_epic = df_ctrl.at[subject, cpg]
        epic_data.append(meth_epic)
        meth_agena = agena.at[subject, cpg]
        agena_data.append(meth_agena)
    pval = pvals_df.at[cpg, 'pvals_fdr_bh']

    epic_data = np.array(epic_data)
    epic_data = epic_data[~np.isnan(epic_data)]

    agena_data = np.array(agena_data)
    agena_data = agena_data[~np.isnan(agena_data)]

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=epic_data,
            name=f"850K",
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
            line_color='black',
            fillcolor='blue',
            marker=dict(color='blue', line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(epic_data) / 25,
            opacity=0.8
        )
    )
    fig.add_trace(
        go.Violin(
            y=agena_data,
            name=f"Agena",
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
            line_color='black',
            fillcolor='red',
            marker=dict(color='red', line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(agena_data) / 25,
            opacity=0.8
        )
    )
    gene = manifest.at[cpg, 'Gene']
    add_layout(fig, "", "Beta value", f"{cpg} ({gene})<br>p-value: {pval:0.2e}")
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_xaxes(tickfont_size=15)
    fig.update_layout(
        margin=go.layout.Margin(
            l=110,
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
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    save_figure(fig, f"{path_save}/figs/cpgs/{cpg_id:3d}_{cpg}")

features_for_clock = pvals_df.loc[pvals_df['pvals_fdr_bh'] > 0.05, :].index.values

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=-np.log10(pvals_corr),
        y=list(range(len(pvals_corr))),
        orientation='h',
        marker=dict(color='red', opacity=0.9)
    )
)
fig.add_trace(
    go.Scatter(
        x=[-np.log10(0.05), -np.log10(0.05)],
        y=[-1, len(pvals_corr)],
        showlegend=False,
        mode='lines',
        line = dict(color='black', width=2, dash='dash')
    )
)
add_layout(fig, "$\\huge{-\log_{10}(\\text{p-value})}$", "", f"")
fig.update_layout({'colorway': ['red', 'black']})
fig.update_layout(legend_font_size=20)
fig.update_layout(showlegend=False)
fig.update_layout(
    yaxis = dict(
        tickmode = 'array',
        tickvals = list(range(len(pvals_corr))),
        ticktext = cpgs_common
    )
)
fig.update_yaxes(autorange=False)
fig.update_layout(yaxis_range=[-1, len(pvals_corr)])
fig.update_yaxes(tickfont_size=24)
fig.update_xaxes(tickfont_size=30)
fig.update_layout(
    autosize=False,
    width=800,
    height=1000,
    margin=go.layout.Margin(
        l=175,
        r=20,
        b=100,
        t=40,
        pad=0
    )
)
save_figure(fig, f"{path_save}/figs/pvals_corr")

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
for train_idx, val_idx in k_fold.split(range(df_ctrl.shape[0])):
    X_train = df_ctrl.loc[df_ctrl.index[train_idx], features_for_clock].to_numpy()
    y_train = df_ctrl.loc[df_ctrl.index[train_idx], target].to_numpy()
    X_val = df_ctrl.loc[df_ctrl.index[val_idx], features_for_clock].to_numpy()
    y_val = df_ctrl.loc[df_ctrl.index[val_idx], target].to_numpy()

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
for f_id, f in enumerate(features_for_clock):
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

y_pred_ctrl_train = calc_metrics(best_model, df_ctrl.loc[df_ctrl.index[best_train_idx], features_for_clock].to_numpy(), df_ctrl.loc[df_ctrl.index[best_train_idx], target].to_numpy(), 'Control_train', params)
y_pred_ctrl_val = calc_metrics(best_model, df_ctrl.loc[df_ctrl.index[best_val_idx], features_for_clock].to_numpy(), df_ctrl.loc[df_ctrl.index[best_val_idx], target].to_numpy(), 'Control_val', params)
y_pred_esrd_val = calc_metrics(best_model, df_case[features_for_clock].to_numpy(), df_case[target].to_numpy(), 'ESRD', params)
y_pred_all = calc_metrics(best_model, df[features_for_clock].to_numpy(), df[target].to_numpy(), 'All', params)
y_pred_agena = calc_metrics(best_model, agena.loc[subjects_common, features_for_clock].to_numpy(),  df_ctrl.loc[subjects_common, target].to_numpy(), 'agena', params)
y_pred_850k = calc_metrics(best_model, df_ctrl.loc[subjects_common, features_for_clock].to_numpy(),  df_ctrl.loc[subjects_common, target].to_numpy(), '850K', params)
params['num_features'] = num_features
params_df = pd.DataFrame({'Feature': list(params.keys()), 'Value': list(params.values())})
params_df.to_excel(f'{path_save}/params.xlsx', index=False)
print(params_df)

ages_df = pd.DataFrame(index=subjects_common)
ages_df['age_estimation_agena'] = y_pred_agena
ages_df['age_estimation_850K'] = y_pred_850k

df['age_estimation'] = y_pred_all
ctrl = df.loc[df['Group'] == 'Control']
ctrl_train = ctrl.loc[ctrl.index[best_train_idx], :]
ctrl_val = ctrl.loc[ctrl.index[best_val_idx], :]
esrd = df.loc[df['Group'] == 'ESRD']
test_epic = df.loc[subjects_common, :]
test_agena = agena.loc[subjects_common, :]
test_agena['age_estimation'] = y_pred_agena
test_agena['Age'] = df.loc[subjects_common, "Age"].values

clock_name = 'age_estimation'

formula = f"{clock_name} ~ Age"
model_linear = smf.ols(formula=formula, data=ctrl_train).fit()
ctrl_train[f"{clock_name}_acceleration"] = ctrl_train[f'{clock_name}'] - model_linear.predict(ctrl_train)
ctrl_val[f"{clock_name}_acceleration"] = ctrl_val[f'{clock_name}'] - model_linear.predict(ctrl_val)
esrd[f"{clock_name}_acceleration"] = esrd[f'{clock_name}'] - model_linear.predict(esrd)
test_epic[f"{clock_name}_acceleration"] = test_epic[f'{clock_name}'] - model_linear.predict(test_epic)
test_agena[f"{clock_name}_acceleration"] = test_agena[f'{clock_name}'] - model_linear.predict(test_agena)

values_test_epic = test_epic.loc[:, f"{clock_name}_acceleration"].values
values_test_agena = test_agena.loc[:, f"{clock_name}_acceleration"].values

stat_aa, pval_aa = mannwhitneyu(values_test_epic, values_test_agena, alternative='two-sided')

fig = go.Figure()
fig.add_trace(
    go.Violin(
        y=values_test_epic,
        name=f"850K",
        box_visible=True,
        meanline_visible=True,
        showlegend=True,
        line_color='black',
        fillcolor="blue",
        marker=dict(color='blue', line=dict(color='black', width=0.3), opacity=0.8),
        points='all',
        bandwidth=np.ptp(values_test_epic) / 20,
        opacity=0.8
    )
)
fig.add_trace(
    go.Violin(
        y=values_test_agena,
        name=f"Agena",
        box_visible=True,
        meanline_visible=True,
        showlegend=True,
        line_color='black',
        fillcolor='red',
        marker=dict(color='red', line=dict(color='black', width=0.3), opacity=0.8),
        points='all',
        bandwidth=np.ptp(values_test_agena) / 20,
        opacity=0.8
    )
)

add_layout(fig, "", "Age estimation acceleration", f"p-value: {pval_aa:0.2e}")
fig.update_layout(title_xref='paper')
fig.update_layout(legend_font_size=20)
fig.update_xaxes(tickfont_size=15)
fig.update_layout(
    margin=go.layout.Margin(
        l=110,
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
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

save_figure(fig, f"{path_save}/figs/age_acceleration")
