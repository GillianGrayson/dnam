import pandas as pd
from scripts.python.routines.manifest import get_manifest
import numpy as np
import os
import pickle
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from scripts.python.routines.plot.scatter import add_scatter_trace
from matplotlib import colors
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.box import add_box_trace
from scripts.python.routines.plot.layout import add_layout
import pathlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from functools import reduce
import plotly
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.histogram import add_histogram_trace
from scripts.python.routines.plot.layout import add_layout
from scripts.python.routines.plot.p_value import add_p_value_annotation


thld_abs_diff = 10000

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

with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    immuno_features = f.read().splitlines()

path_save = f"{path}/{platform}/{dataset}/special/011_immuno_part3_and_part4_check_clocks"
pathlib.Path(f"{path_save}/figs/{thld_abs_diff}").mkdir(parents=True, exist_ok=True)

continuous_vars = {}
categorical_vars = {status_col: [x.column for x in status_passed_fields], sex_col: list(sex_dict.values())}
pheno = pd.read_pickle(f"{path}/{platform}/{dataset}/pheno_xtd.pkl")
pheno = filter_pheno(dataset, pheno, continuous_vars, categorical_vars)
ctrl = pheno.loc[pheno['Group'] == 'Control']
esrd = pheno.loc[pheno['Group'] == 'ESRD']

formula = f"ImmunoAge ~ Age"
model_linear = smf.ols(formula=formula, data=ctrl).fit()
y_pred = model_linear.predict(pheno)
pheno['ImmunoAgeAcc'] = pheno['ImmunoAge'] - y_pred
ImmunoAgeAcc_check = abs(pheno['ImmunoAgeAcc'] - pheno['ImmunoAgeAA'])
print(f"Linear model checking: {max(abs(ImmunoAgeAcc_check.values))}")

df = pd.read_excel(f"{path}/{platform}/{dataset}/data/immuno/part3_part4_with_age_sex.xlsx", index_col='ID')

model_df = pd.read_excel(f"{path}/{platform}/{dataset}/special/011_immuno_part3_and_part4_check_clocks/legacy/Control/v22/clock.xlsx")

features = model_df['feature'].to_list()
coefs = model_df['coef'].to_list()

# Checking of model
immuno_check = np.full(pheno.shape[0], coefs[0])
for feat_id in range(1, len(features)):
    immuno_check += pheno.loc[:, features[feat_id]].values * coefs[feat_id]
check = immuno_check - pheno.loc[:, 'ImmunoAge'].values
print(f"Clock model checking: {max(abs(check))}")

predicted = np.full(df.shape[0], coefs[0])
for feat_id in range(1, len(features)):
    predicted += df.loc[:, features[feat_id]].values * coefs[feat_id]

df[f'ImmunoAge'] = predicted
df.to_excel(f"{path}/{platform}/{dataset}/data/immuno/part3_part4_with_age_sex.xlsx", index=True)

df[f'ImmunoAgeAbsDiff'] = df[f'ImmunoAge'] - df[f'Age']
df[f'Name'] = df.index.values
y_pred = model_linear.predict(df)
df['ImmunoAgeAcc'] = df['ImmunoAge'] - y_pred
df = df[~df.index.str.startswith(('Q', 'H'))]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df.loc[abs(df[f'ImmunoAgeAbsDiff']) <= thld_abs_diff, 'Age'].values,
        y=df.loc[abs(df[f'ImmunoAgeAbsDiff']) <= thld_abs_diff, 'ImmunoAge'].values,
        showlegend=True,
        name="Controls",
        mode="markers",
        marker=dict(
            size=9,
            opacity=0.65,
            line=dict(
                width=0.5
            )
        )
    )
)
fig.add_trace(
    go.Scatter(
        x=df.loc[abs(df[f'ImmunoAgeAbsDiff']) > thld_abs_diff, 'Age'].values,
        y=df.loc[abs(df[f'ImmunoAgeAbsDiff']) > thld_abs_diff, 'ImmunoAge'].values,
        showlegend=False,
        mode="markers+text",
        text=df.loc[abs(df[f'ImmunoAgeAbsDiff']) > thld_abs_diff, 'Name'].values,
        textposition="middle right",
        marker=dict(
            size=10,
            opacity=0.65,
            line=dict(
                width=1
            )
        )
    )
)
add_layout(fig, 'Age', 'ipAGE', f"")
fig.update_layout({'colorway': ["blue", "blue", "red"]})
fig.update_layout(
    margin=go.layout.Margin(
        l=80,
        r=20,
        b=80,
        t=50,
        pad=0
    )
)
save_figure(fig, f"{path_save}/figs/{thld_abs_diff}/x(Age)_y(ImmunoAge)")

df = df[abs(df[f'ImmunoAgeAbsDiff']) <= thld_abs_diff]
print(f"Shape: {df.shape}")
df.to_excel(f"{path_save}/part3_part4_filtered_with_age_sex_{thld_abs_diff}.xlsx", index=True)
rmse = np.sqrt(mean_squared_error(df.loc[:, 'Age'].values, df.loc[:, 'ImmunoAge'].values))
mae = mean_absolute_error(df.loc[:, 'Age'].values, df.loc[:, 'ImmunoAge'].values)
print(f"RMSE in test controls: {rmse}")
print(f"MAE in test controls: {mae}")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df.loc[:, 'Age'].values,
        y=df.loc[:, 'ImmunoAge'].values,
        showlegend=True,
        name="Controls",
        mode="markers",
        marker=dict(
            size=9,
            opacity=0.65,
            line=dict(
                width=0.5
            )
        )
    )
)
add_layout(fig, 'Age', 'ipAGE', f"")
fig.update_layout({'colorway': ["blue", "blue", "red"]})
fig.update_layout(
    margin=go.layout.Margin(
        l=80,
        r=20,
        b=80,
        t=50,
        pad=0
    )
)
save_figure(fig, f"{path_save}/figs/{thld_abs_diff}/x(Age)_y(ImmunoAge)_filtered")

ctrl_color = 'lime'
ctrl_test_color = 'cyan'
esrd_color = 'fuchsia'
dist_num_bins = 25
ctrl = pheno.loc[pheno['Group'] == 'Control']
esrd = pheno.loc[pheno['Group'] == 'ESRD']
ctrl_test = df

values_ctrl = ctrl.loc[:, 'ImmunoAgeAA'].values
values_ctrl_test = ctrl_test.loc[:, 'ImmunoAgeAcc'].values
values_esrd = esrd.loc[:, 'ImmunoAgeAA'].values

stat_01, pval_01 = mannwhitneyu(values_ctrl, values_ctrl_test, alternative='two-sided')
stat_02, pval_02 = mannwhitneyu(values_ctrl, values_esrd, alternative='two-sided')
stat_12, pval_12 = mannwhitneyu(values_ctrl_test, values_esrd, alternative='two-sided')

fig = go.Figure()
add_box_trace(fig, pheno.loc[pheno['Group'] == 'Control', 'ImmunoAgeAcc'].values, f"Control (model building)")
add_box_trace(fig, df.loc[:, 'ImmunoAgeAcc'].values, f"Control (test)")
add_box_trace(fig, pheno.loc[pheno['Group'] == 'ESRD', 'ImmunoAgeAcc'].values, f"ESRD")
add_layout(fig, "", "Age acceleration", f"")
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.25,
        xanchor="center",
        x=0.5
    )
)
fig.update_layout({'colorway': ['blue', 'cyan', 'red']})
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(autorange=False)
fig.update_layout(yaxis_range=[-50, 250])
fig = add_p_value_annotation(fig, {(0,1): pval_01, (1, 2) : pval_12, (0,2): pval_02})
fig.update_layout(
    margin=go.layout.Margin(
        l=80,
        r=20,
        b=30,
        t=120,
        pad=0
    )
)

save_figure(fig, f"{path_save}/figs/{thld_abs_diff}/box_age_acceleration")

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
        y=values_ctrl_test,
        name=f"Control (test)",
        box_visible=True,
        meanline_visible=True,
        showlegend=True,
        line_color='black',
        fillcolor=ctrl_test_color,
        marker=dict(color=ctrl_test_color, line=dict(color='black', width=0.3), opacity=0.8),
        points='all',
        bandwidth=np.ptp(values_ctrl_test) / dist_num_bins,
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

add_layout(fig, "", "ipAGE acceleration", f"")
fig.update_layout({'colorway': ['lime', 'cyan', 'fuchsia']})
fig = add_p_value_annotation(fig, {(0,1): pval_01, (1, 2) : pval_12, (0,2): pval_02})
fig.update_yaxes(autorange=False)
fig.update_layout(yaxis_range=[-50, 200])
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
save_figure(fig, f"{path_save}/figs/{thld_abs_diff}/vio_age_acceleration")

fig = go.Figure()
add_histogram_trace(fig, df.loc[df['Sex'] == 'M', 'Age'].values, f"Males ({df.loc[df['Sex'] == 'M', :].shape[0]})", 5.0)
add_histogram_trace(fig, df.loc[df['Sex'] == 'F', 'Age'].values, f"Females ({df.loc[df['Sex'] == 'F', :].shape[0]})", 5.0)
add_layout(fig, "Age", "Count", "")
fig.update_layout(colorway=['blue', 'red'], barmode='overlay')
fig.update_layout(
    margin=go.layout.Margin(
        l=50,
        r=10,
        b=60,
        t=40,
        pad=0
    )
)
save_figure(fig, f"{path_save}/figs/{thld_abs_diff}/histogram_Age")

fig = go.Figure()
x = df.loc[:, 'Age'].values,
y = df.loc[:, 'ImmunoAge'].values,
add_scatter_trace(fig, ctrl.loc[:, 'Age'].values, ctrl.loc[:, 'ImmunoAge'].values, f"Control (original)")
add_scatter_trace(fig, ctrl.loc[:, 'Age'].values, model_linear.fittedvalues.values, "", "lines")
add_scatter_trace(fig, df.loc[:, 'Age'].values, df.loc[:, 'ImmunoAge'].values, f"Control (test)")
add_scatter_trace(fig, esrd.loc[:, 'Age'].values, esrd.loc[:, 'ImmunoAge'].values, f"ESRD")
add_layout(fig, f"Age", 'ipAGE', f"")
fig.update_layout({'colorway': ['blue', 'blue', 'cyan', 'red']})
fig.update_layout(
    margin=go.layout.Margin(
        l=80,
        r=20,
        b=80,
        t=50,
        pad=0
    )
)
fig.update_yaxes(autorange=False)
fig.update_xaxes(autorange=False)
fig.update_layout(yaxis_range=[10, 100])
fig.update_layout(xaxis_range=[10, 100])
save_figure(fig, f"{path_save}/figs/{thld_abs_diff}/scatter_Age_ipAGE")