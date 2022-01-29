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
from scipy.stats import chi2_contingency
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scripts.python.pheno.datasets.filter import filter_pheno
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from scripts.python.routines.plot.scatter import add_scatter_trace
import string
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
import pathlib
import plotly.io as pio
pio.kaleido.scope.mathjax = None
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout, get_axis
from scripts.python.routines.plot.p_value import add_p_value_annotation


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

is_title_pval = True
is_percentage = False

path_save = f"{path}/{platform}/{dataset}/special/015_new_pheno"
pathlib.Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

pheno.set_index("ID", inplace=True, verify_integrity=True)

pheno_new = pd.read_excel(f"{path}/{platform}/{dataset}/data/pheno/pheno_new_28_01_22.xlsx", index_col='ID')
missed_ids = list(set(pheno.index.values) - set(pheno_new.index.values))
print(f"Missed IDs: {'_'.join(missed_ids)}")
duplicates = pheno_new.index[pheno_new.index.duplicated()]
print(f"Duplicates IDs: {'_'.join(missed_ids)}")
pheno_new.fillna('No info', inplace=True)

target_features = pheno_new.columns.values

df = pd.merge(pheno, pheno_new, left_index=True, right_index=True)
df.to_excel(f"{path_save}/pheno_xtd.xlsx", index=True)
num_subjects = df.shape[0]
df_ctrl = df.loc[(df[status_col] == 'Control'), :]
df_esrd = df.loc[(df[status_col] == 'ESRD'), :]

ctrl_color = 'lime'
esrd_color = 'fuchsia'

variants = {
    'Smoking': ['No', 'Yes', 'No info'],
    'Insomnia': ['No', 'Yes', 'No info'],
    'Regular nutrition': ['No', 'Yes'],
    'Alcohol consumption': ['No', 'Yes', 'No info'],
    'Marital status': ['Single', 'Married', 'Divorced', 'Widow(er)', 'No info'],
    'Currently working': ['No', 'Yes'],
    'Accommodation': ['City', 'Village'],
    'Education level': ['Higher', 'Higher unfinished', 'Secondary', 'Secondary special', 'Secondary unfinished', 'No info'],
    'Work type (intellectual or physical)': ['Intellectual', 'Physical', 'No info']
}

variants_x_tick_font_size = {
    'Smoking': 25,
    'Insomnia': 25,
    'Regular nutrition': 25,
    'Alcohol consumption': 25,
    'Marital status': 20,
    'Currently working': 25,
    'Accommodation': 25,
    'Education level': 15,
    'Work type (intellectual or physical)': 25
}

for f_id, f in enumerate(target_features):
    xs = variants[f]
    cat_df = pd.DataFrame(index=['Control', 'ESRD'], columns=xs)
    ys_ctrl = []
    ys_esrd = []
    for x in xs:
        if is_percentage:
            y_ctrl = df_ctrl[df_ctrl[f] == x].shape[0] / num_subjects * 100
            y_esrd = df_esrd[df_esrd[f] == x].shape[0] / num_subjects * 100
        else:
            y_ctrl = df_ctrl[df_ctrl[f] == x].shape[0]
            y_esrd = df_esrd[df_esrd[f] == x].shape[0]
        ys_ctrl.append(y_ctrl)
        ys_esrd.append(y_esrd)
        cat_df.loc['Control', x] = y_ctrl
        cat_df.loc['ESRD', x] = y_esrd

    chi2, p, dof, ex = chi2_contingency(cat_df)

    if f == 'Education level':
        xs = [x.replace(' ', '<br>') for x in xs]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name='Control',
            x=xs,
            y=ys_ctrl,
            text=ys_ctrl,
            textposition='auto',
            orientation='v',
            marker=dict(color=ctrl_color, opacity=0.9)
        )
    )
    fig.add_trace(
        go.Bar(
            name='ESRD',
            x=xs,
            y=ys_esrd,
            text=ys_esrd,
            textposition='auto',
            orientation='v',
            marker=dict(color=esrd_color, opacity=0.9)
        )
    )
    fig.update_layout(barmode='group')
    if is_title_pval:
        if is_percentage:
            add_layout(fig, f"{f}", "% of participants", f"Chi-square p-value = {p:0.2e}")
        else:
            add_layout(fig, f"{f}", "Number of participants", f"Chi-square p-value = {p:0.2e}")
        fig.update_layout(title_xref='paper')
        fig.update_layout(
            autosize=False,
            margin=go.layout.Margin(
                l=100,
                r=20,
                b=100,
                t=90,
                pad=0
            )
        )
        fig.add_annotation(dict(font=dict(color='black', size=45),
                                x=-0.18,
                                y=1.31,
                                showarrow=False,
                                text=f"({string.ascii_lowercase[f_id]})",
                                textangle=0,
                                yanchor='top',
                                xanchor='left',
                                xref="paper",
                                yref="paper"))
    else:
        if is_percentage:
            add_layout(fig, f"{f}", "% of participants", f"")
        else:
            add_layout(fig, f"{f}", "Number of participants", f"")
        fig.update_layout(
            autosize=False,
            margin=go.layout.Margin(
                l=100,
                r=20,
                b=100,
                t=40,
                pad=0
            )
        )
        fig.add_annotation(dict(font=dict(color='black', size=45),
                                x=-0.18,
                                y=1.18,
                                showarrow=False,
                                text=f"({string.ascii_lowercase[f_id]})",
                                textangle=0,
                                yanchor='top',
                                xanchor='left',
                                xref="paper",
                                yref="paper"))
    fig.update_layout(legend_font_size=20)
    fig.update_layout(showlegend=True)

    fig.update_traces(textposition='auto')
    if is_percentage:
        fig.update_traces(texttemplate='%{text:.1f}')
    fig.update_xaxes(tickfont_size=variants_x_tick_font_size[f])
    fig.update_traces(textfont_size=21, selector=dict(type='bar'))
    fig.update_traces(marker_line_color='black', marker_line_width=1.5)
    save_figure(fig, f"{path_save}/{string.ascii_lowercase[f_id]}_{f}")
