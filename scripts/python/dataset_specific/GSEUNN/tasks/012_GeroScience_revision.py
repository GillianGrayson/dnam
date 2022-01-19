import pandas as pd
from scripts.python.routines.manifest import get_manifest
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from plotly.subplots import make_subplots
import os
import pickle
import upsetplot as upset
from upsetplot import UpSet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scripts.python.pheno.datasets.filter import filter_pheno
import matplotlib.pyplot as plt
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from scripts.python.routines.plot.scatter import add_scatter_trace
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
from scripts.python.routines.sections import get_sections
import scripts.python.routines.plot.venn as vennrout
from scipy import stats
import pathlib
import string
import statsmodels.formula.api as smf
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout, get_axis
from scripts.python.routines.plot.p_value import add_p_value_annotation
from scripts.python.EWAS.routines.correction import correct_pvalues
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
ctrl = pheno.loc[pheno['Group'] == 'Control']
esrd = pheno.loc[pheno['Group'] == 'ESRD']

ctrl_color = 'lime'
ctrl_test_color = 'cyan'
esrd_color = 'fuchsia'
dist_num_bins = 25

path_save = f"{path}/{platform}/{dataset}/special/012_GeroScience_revision"
pathlib.Path(f"{path_save}/SupplementaryFigure2").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/SupplementaryFigure3").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/SupplementaryFigure4").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/SupplementaryFigure5").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/SupplementaryFigure6").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/SupplementaryFigure7").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/Figure2").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/Figure3").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/Figure4").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/Figure5").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/Figure6").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/Figure7").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{path_save}/SupplementaryTable9").mkdir(parents=True, exist_ok=True)

# Supplementary Figure 4 ===============================================================================================
with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    features = f.read().splitlines()

num_cols = 5
num_rows = int(np.ceil(len(features) / num_cols))

fig = make_subplots(rows=num_rows, cols=num_cols, shared_yaxes=False)

for r_id in range(num_rows):
    for c_id in range(num_cols):
        rc_id = r_id * num_cols + c_id
        if rc_id < len(features):
            f = features[rc_id]

            if rc_id == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[1000],
                        y=[0],
                        showlegend=True,
                        name='Control',
                        mode='markers',
                        marker=dict(
                            size=100,
                            opacity=0.7,
                            line=dict(
                                width=0.1
                            ),
                            color='lime'
                        )
                    ),
                    row=r_id + 1,
                    col=c_id + 1
                )
                fig.add_trace(
                    go.Scatter(
                        x=[1000],
                        y=[0],
                        showlegend=True,
                        name='ESRD',
                        mode='markers',
                        marker=dict(
                            size=100,
                            opacity=0.7,
                            line=dict(
                                width=0.1
                            ),
                            color='fuchsia'
                        )
                    ),
                    row=r_id + 1,
                    col=c_id + 1
                )

            x_ctrl = ctrl.loc[:, 'Age'].values
            y_ctrl = ctrl.loc[:, f].values
            x_esrd = esrd.loc[:, 'Age'].values
            y_esrd = esrd.loc[:, f].values

            y_s = 0
            y_pctl = np.percentile(pheno.loc[:, f].values, [90])[0]
            y_max = np.max(pheno.loc[:, f].values)
            if y_max > 2*y_pctl:
                y_f = y_pctl * 1.3
            else:
                y_f = y_max

            fig.add_trace(
                go.Scatter(
                    x=x_ctrl,
                    y=y_ctrl,
                    showlegend=False,
                    name='Control',
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.7,
                        line=dict(
                            width=0.1
                        ),
                        color='lime'
                    )
                ),
                row=r_id + 1,
                col=c_id + 1
            )
            fig.add_trace(
                go.Scatter(
                    x=x_esrd,
                    y=y_esrd,
                    showlegend=False,
                    name='ESRD',
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.7,
                        line=dict(
                            width=0.1
                        ),
                        color='fuchsia'
                    )
                ),
                row=r_id + 1,
                col=c_id + 1
            )
            fig.update_xaxes(
                autorange=False,
                title_text="Age",
                range=[10, 100],
                row=r_id + 1,
                col=c_id + 1,
                showgrid=True,
                zeroline=False,
                linecolor='black',
                showline=True,
                gridcolor='gainsboro',
                gridwidth=0.05,
                mirror=True,
                ticks='outside',
                titlefont=dict(
                    color='black',
                    size=20
                ),
                showticklabels=True,
                tickangle=0,
                tickfont=dict(
                    color='black',
                    size=20
                ),
                exponentformat='e',
                showexponent='all'
            )
            fig.update_yaxes(
                autorange=False,
                title_text=f"{f}",
                range=[y_s, y_f],
                row=r_id + 1,
                col=c_id + 1,
                showgrid=True,
                zeroline=False,
                linecolor='black',
                showline=True,
                gridcolor='gainsboro',
                gridwidth=0.05,
                mirror=True,
                ticks='outside',
                titlefont=dict(
                    color='black',
                    size=20
                ),
                showticklabels=True,
                tickangle=0,
                tickfont=dict(
                    color='black',
                    size=20
                ),
                exponentformat='e',
                showexponent='all'
            )

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="center",
        x=0.5
    ),
    title=dict(
        text="",
        font=dict(size=25)
    ),
    template="none",
    autosize=False,
    width=3000,
    height=4000,
    margin=go.layout.Margin(
        l=100,
        r=40,
        b=100,
        t=100,
        pad=0
    )
)
fig.update_layout(legend_font_size=50)
save_figure(fig, f"{path_save}/SupplementaryFigure4/tmp")

# Supplementary Figure 5 ===============================================================================================
with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    features = f.read().splitlines()

num_cols = 5
num_rows = int(np.ceil(len(features) / num_cols))

fig = make_subplots(rows=num_rows, cols=num_cols, shared_yaxes=False)

for r_id in range(num_rows):
    for c_id in range(num_cols):
        rc_id = r_id * num_cols + c_id
        if rc_id < len(features):
            f = features[rc_id]

            if rc_id == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[1000],
                        y=[0],
                        showlegend=True,
                        name='Control',
                        mode='markers',
                        marker=dict(
                            size=100,
                            opacity=0.7,
                            line=dict(
                                width=0.1
                            ),
                            color='lime'
                        )
                    ),
                    row=r_id + 1,
                    col=c_id + 1
                )
                fig.add_trace(
                    go.Scatter(
                        x=[1000],
                        y=[0],
                        showlegend=True,
                        name='ESRD',
                        mode='markers',
                        marker=dict(
                            size=100,
                            opacity=0.7,
                            line=dict(
                                width=0.1
                            ),
                            color='fuchsia'
                        )
                    ),
                    row=r_id + 1,
                    col=c_id + 1
                )

            x_ctrl = ctrl.loc[:, 'ImmunoAge'].values
            y_ctrl = ctrl.loc[:, f].values
            x_esrd = esrd.loc[:, 'ImmunoAge'].values
            y_esrd = esrd.loc[:, f].values

            y_s = 0
            y_pctl = np.percentile(pheno.loc[:, f].values, [90])[0]
            y_max = np.max(pheno.loc[:, f].values)
            if y_max > 2*y_pctl:
                y_f = y_pctl * 1.3
            else:
                y_f = y_max

            fig.add_trace(
                go.Scatter(
                    x=x_ctrl,
                    y=y_ctrl,
                    showlegend=False,
                    name='Control',
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.7,
                        line=dict(
                            width=0.1
                        ),
                        color='lime'
                    )
                ),
                row=r_id + 1,
                col=c_id + 1
            )
            fig.add_trace(
                go.Scatter(
                    x=x_esrd,
                    y=y_esrd,
                    showlegend=False,
                    name='ESRD',
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.7,
                        line=dict(
                            width=0.1
                        ),
                        color='fuchsia'
                    )
                ),
                row=r_id + 1,
                col=c_id + 1
            )
            fig.update_xaxes(
                autorange=False,
                title_text="ipAGE",
                range=[0, 250],
                row=r_id + 1,
                col=c_id + 1,
                showgrid=True,
                zeroline=False,
                linecolor='black',
                showline=True,
                gridcolor='gainsboro',
                gridwidth=0.05,
                mirror=True,
                ticks='outside',
                titlefont=dict(
                    color='black',
                    size=20
                ),
                showticklabels=True,
                tickangle=0,
                tickfont=dict(
                    color='black',
                    size=20
                ),
                exponentformat='e',
                showexponent='all'
            )
            fig.update_yaxes(
                autorange=False,
                title_text=f"{f}",
                range=[y_s, y_f],
                row=r_id + 1,
                col=c_id + 1,
                showgrid=True,
                zeroline=False,
                linecolor='black',
                showline=True,
                gridcolor='gainsboro',
                gridwidth=0.05,
                mirror=True,
                ticks='outside',
                titlefont=dict(
                    color='black',
                    size=20
                ),
                showticklabels=True,
                tickangle=0,
                tickfont=dict(
                    color='black',
                    size=20
                ),
                exponentformat='e',
                showexponent='all'
            )

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="center",
        x=0.5
    ),
    title=dict(
        text="",
        font=dict(size=25)
    ),
    template="none",
    autosize=False,
    width=3000,
    height=4000,
    margin=go.layout.Margin(
        l=100,
        r=40,
        b=100,
        t=100,
        pad=0
    )
)
fig.update_layout(legend_font_size=50)
save_figure(fig, f"{path_save}/SupplementaryFigure5/tmp")

# Supplementary Figure 5 ===============================================================================================
with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    features = f.read().splitlines()

num_cols = 5
num_rows = int(np.ceil(len(features) / num_cols))

fig = make_subplots(rows=num_rows, cols=num_cols, shared_yaxes=False)

for r_id in range(num_rows):
    for c_id in range(num_cols):
        rc_id = r_id * num_cols + c_id
        if rc_id < len(features):
            f = features[rc_id]

            if rc_id == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[1000],
                        y=[0],
                        showlegend=True,
                        name='Control',
                        mode='markers',
                        marker=dict(
                            size=100,
                            opacity=0.7,
                            line=dict(
                                width=0.1
                            ),
                            color='lime'
                        )
                    ),
                    row=r_id + 1,
                    col=c_id + 1
                )
                fig.add_trace(
                    go.Scatter(
                        x=[1000],
                        y=[0],
                        showlegend=True,
                        name='ESRD',
                        mode='markers',
                        marker=dict(
                            size=100,
                            opacity=0.7,
                            line=dict(
                                width=0.1
                            ),
                            color='fuchsia'
                        )
                    ),
                    row=r_id + 1,
                    col=c_id + 1
                )

            x_ctrl = ctrl.loc[:, 'ImmunoAgeAA'].values
            y_ctrl = ctrl.loc[:, f].values
            x_esrd = esrd.loc[:, 'ImmunoAgeAA'].values
            y_esrd = esrd.loc[:, f].values

            y_s = 0
            y_pctl = np.percentile(pheno.loc[:, f].values, [90])[0]
            y_max = np.max(pheno.loc[:, f].values)
            if y_max > 2*y_pctl:
                y_f = y_pctl * 1.3
            else:
                y_f = y_max

            fig.add_trace(
                go.Scatter(
                    x=x_ctrl,
                    y=y_ctrl,
                    showlegend=False,
                    name='Control',
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.7,
                        line=dict(
                            width=0.1
                        ),
                        color='lime'
                    )
                ),
                row=r_id + 1,
                col=c_id + 1
            )
            fig.add_trace(
                go.Scatter(
                    x=x_esrd,
                    y=y_esrd,
                    showlegend=False,
                    name='ESRD',
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.7,
                        line=dict(
                            width=0.1
                        ),
                        color='fuchsia'
                    )
                ),
                row=r_id + 1,
                col=c_id + 1
            )
            fig.update_xaxes(
                autorange=False,
                title_text="ipAGE acceleration",
                range=[-50, 200],
                row=r_id + 1,
                col=c_id + 1,
                showgrid=True,
                zeroline=False,
                linecolor='black',
                showline=True,
                gridcolor='gainsboro',
                gridwidth=0.05,
                mirror=True,
                ticks='outside',
                titlefont=dict(
                    color='black',
                    size=20
                ),
                showticklabels=True,
                tickangle=0,
                tickfont=dict(
                    color='black',
                    size=20
                ),
                exponentformat='e',
                showexponent='all'
            )
            fig.update_yaxes(
                autorange=False,
                title_text=f"{f}",
                range=[y_s, y_f],
                row=r_id + 1,
                col=c_id + 1,
                showgrid=True,
                zeroline=False,
                linecolor='black',
                showline=True,
                gridcolor='gainsboro',
                gridwidth=0.05,
                mirror=True,
                ticks='outside',
                titlefont=dict(
                    color='black',
                    size=20
                ),
                showticklabels=True,
                tickangle=0,
                tickfont=dict(
                    color='black',
                    size=20
                ),
                exponentformat='e',
                showexponent='all'
            )

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="center",
        x=0.5
    ),
    title=dict(
        text="",
        font=dict(size=25)
    ),
    template="none",
    autosize=False,
    width=3000,
    height=4000,
    margin=go.layout.Margin(
        l=100,
        r=40,
        b=100,
        t=100,
        pad=0
    )
)
fig.update_layout(legend_font_size=50)
save_figure(fig, f"{path_save}/SupplementaryFigure6/tmp")

# Supplementary Figure 7 ===============================================================================================
with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    features= f.read().splitlines()
result_dict = {
    'feature': features,
    'name': features
}

corr_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])
pval_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])
corr_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])
pval_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])

for f_id_1, f_1 in enumerate(result_dict['feature']):
    for f_id_2, f_2 in enumerate(result_dict['feature']):
        values_1_ctrl = ctrl.loc[:, f_1].values
        values_2_ctrl = ctrl.loc[:, f_2].values
        values_1_esrd = esrd.loc[:, f_1].values
        values_2_esrd = esrd.loc[:, f_2].values

        corr_ctrl, pval_ctrl = stats.pearsonr(values_1_ctrl, values_2_ctrl)
        corr_df_ctrl.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = corr_ctrl
        pval_df_ctrl.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = pval_ctrl
        corr_esrd, pval_esrd = stats.pearsonr(values_1_esrd, values_2_esrd)
        corr_df_esrd.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = corr_esrd
        pval_df_esrd.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = pval_esrd

selection = np.tri(pval_df_ctrl.shape[0], pval_df_ctrl.shape[1], -1, dtype=np.bool)

pval_df_ctrl_fdr = pval_df_ctrl.where(selection).stack().reset_index()
pval_df_ctrl_fdr.columns = ['row', 'col', 'pval']
_, pvals_corr, _, _ = multipletests(pval_df_ctrl_fdr.loc[:, 'pval'].values, 0.05, method='fdr_bh')
pval_df_ctrl_fdr['pval_fdr_bh'] = pvals_corr

pval_df_esrd_fdr = pval_df_esrd.where(selection).stack().reset_index()
pval_df_esrd_fdr.columns = ['row', 'col', 'pval']
_, pvals_corr, _, _ = multipletests(pval_df_esrd_fdr.loc[:, 'pval'].values, 0.05, method='fdr_bh')
pval_df_esrd_fdr['pval_fdr_bh'] = pvals_corr

for line_id in range(pval_df_ctrl_fdr.shape[0]):
    pval_df_ctrl.loc[pval_df_ctrl_fdr.at[line_id, 'row'], pval_df_ctrl_fdr.at[line_id, 'col']] = pval_df_ctrl_fdr.at[line_id, 'pval_fdr_bh']
    pval_df_ctrl.loc[pval_df_ctrl_fdr.at[line_id, 'col'], pval_df_ctrl_fdr.at[line_id, 'row']] = pval_df_ctrl_fdr.at[line_id, 'pval_fdr_bh']
    pval_df_esrd.loc[pval_df_esrd_fdr.at[line_id, 'row'], pval_df_esrd_fdr.at[line_id, 'col']] = pval_df_esrd_fdr.at[line_id, 'pval_fdr_bh']
    pval_df_esrd.loc[pval_df_esrd_fdr.at[line_id, 'col'], pval_df_esrd_fdr.at[line_id, 'row']] = pval_df_esrd_fdr.at[line_id, 'pval_fdr_bh']

for f_id_1, f_1 in enumerate(result_dict['feature']):
    pval_df_ctrl.loc[result_dict['name'][f_id_1], :] = -np.log10(pval_df_ctrl.loc[result_dict['name'][f_id_1], :])
    pval_df_esrd.loc[result_dict['name'][f_id_1], :] = -np.log10(pval_df_esrd.loc[result_dict['name'][f_id_1], :])

corr_df_ctrl = corr_df_ctrl.iloc[::-1]
mtx_to_plot = corr_df_ctrl.to_numpy()
cmap = plt.get_cmap("bwr").copy()
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-1, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="2%", pad=-1.2)
cbar = ax.figure.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
cax.xaxis.set_label_position('top')
cax.xaxis.set_ticks_position('top')
cbar.set_label(r"$\mathrm{Correlation}$", horizontalalignment='center', fontsize=10)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(corr_df_ctrl.shape[0]))
ax.set_yticks(np.arange(corr_df_ctrl.shape[0]))
ax.set_xticklabels(corr_df_ctrl.columns.values)
ax.set_yticklabels(corr_df_ctrl.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=4)
ax.tick_params(axis='both', which='minor', labelsize=4)
textcolors = ("black", "white")
for i in range(corr_df_ctrl.shape[0]):
    for j in range(corr_df_ctrl.shape[0]):
        color = 'black'
        text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=2)
fig.tight_layout()
plt.savefig(f"{path_save}/SupplementaryFigure7/a_1.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/SupplementaryFigure7/a_1.pdf", bbox_inches='tight', dpi=400)
plt.clf()

pval_df_ctrl = pval_df_ctrl.iloc[::-1]
mtx_to_plot = pval_df_ctrl.to_numpy()
cmap = plt.get_cmap("Reds").copy()
cmap.set_under('lightseagreen')
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="2%", pad=-1.2)
cbar = ax.figure.colorbar(im, cax=cax, orientation="horizontal")
cax.xaxis.set_label_position('top')
cax.xaxis.set_ticks_position('top')
cbar.set_label(r"$-\log_{10}(\mathrm{p-value})$", horizontalalignment='center', fontsize=10)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(pval_df_ctrl.shape[0]))
ax.set_yticks(np.arange(pval_df_ctrl.shape[0]))
ax.set_xticklabels(pval_df_ctrl.columns.values)
ax.set_yticklabels(pval_df_ctrl.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=4)
ax.tick_params(axis='both', which='minor', labelsize=4)
textcolors = ("black", "white")
for i in range(pval_df_ctrl.shape[0]):
    for j in range(pval_df_ctrl.shape[0]):
        color = textcolors[int(im.norm(data[i, j]) > threshold)]
        if np.isinf(mtx_to_plot[i, j]):
            text = ax.text(j, i, f"", ha="center", va="center", color=color, fontsize=2)
        else:
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=2)
fig.tight_layout()
plt.savefig(f"{path_save}/SupplementaryFigure7/a_2.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/SupplementaryFigure7/a_2.pdf", bbox_inches='tight', dpi=400)
plt.clf()

corr_df_esrd = corr_df_esrd.iloc[::-1]
mtx_to_plot = corr_df_esrd.to_numpy()
cmap = plt.get_cmap("bwr").copy()
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-1, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="2%", pad=-1.2)
cbar = ax.figure.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
cax.xaxis.set_label_position('top')
cax.xaxis.set_ticks_position('top')
cbar.set_label(r"$\mathrm{Correlation}$", horizontalalignment='center', fontsize=10)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(corr_df_esrd.shape[0]))
ax.set_yticks(np.arange(corr_df_esrd.shape[0]))
ax.set_xticklabels(corr_df_esrd.columns.values)
ax.set_yticklabels(corr_df_esrd.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=4)
ax.tick_params(axis='both', which='minor', labelsize=4)
textcolors = ("black", "white")
for i in range(corr_df_esrd.shape[0]):
    for j in range(corr_df_esrd.shape[0]):
        color = 'black'
        text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=2)
fig.tight_layout()
plt.savefig(f"{path_save}/SupplementaryFigure7/b_1.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/SupplementaryFigure7/b_1.pdf", bbox_inches='tight', dpi=400)
plt.clf()

pval_df_esrd = pval_df_esrd.iloc[::-1]
mtx_to_plot = pval_df_esrd.to_numpy()
cmap = plt.get_cmap("Reds").copy()
cmap.set_under('lightseagreen')
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="2%", pad=-1.2)
cbar = ax.figure.colorbar(im, cax=cax, orientation="horizontal")
cax.xaxis.set_label_position('top')
cax.xaxis.set_ticks_position('top')
cbar.set_label(r"$-\log_{10}(\mathrm{p-value})$", horizontalalignment='center', fontsize=10)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(pval_df_esrd.shape[0]))
ax.set_yticks(np.arange(pval_df_esrd.shape[0]))
ax.set_xticklabels(pval_df_esrd.columns.values)
ax.set_yticklabels(pval_df_esrd.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=4)
ax.tick_params(axis='both', which='minor', labelsize=4)
textcolors = ("black", "white")
for i in range(pval_df_esrd.shape[0]):
    for j in range(pval_df_esrd.shape[0]):
        color = textcolors[int(im.norm(data[i, j]) > threshold)]
        if np.isinf(mtx_to_plot[i, j]):
            text = ax.text(j, i, f"", ha="center", va="center", color=color, fontsize=2)
        else:
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=2)
fig.tight_layout()
plt.savefig(f"{path_save}/SupplementaryFigure7/b_2.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/SupplementaryFigure7/b_2.pdf", bbox_inches='tight', dpi=400)
plt.clf()

# Supplementary Figure 2 ===============================================================================================
result_dict = {'feature': ['CD4T', 'CD8naive', 'CD8pCD28nCD45RAn', 'Gran', 'Mono', 'NK', 'PlasmaBlast']}
result_dict['pval_mv'] = np.zeros(len(result_dict['feature']))
for f_id, f in enumerate(result_dict['feature']):
    values_ctrl = ctrl.loc[:, f].values
    values_esrd = esrd.loc[:, f].values
    stat, pval = mannwhitneyu(values_ctrl, values_esrd, alternative='two-sided')
    result_dict['pval_mv'][f_id] = pval

result_dict = correct_pvalues(result_dict, ['pval_mv'])
result_df = pd.DataFrame(result_dict)
result_df.set_index('feature', inplace=True)
result_df.sort_values(['pval_mv'], ascending=[True], inplace=True)
result_df.to_excel(f"{path_save}/SupplementaryFigure2/result.xlsx", index=True)

for f_id, f in enumerate(result_dict['feature']):
    values_ctrl = ctrl.loc[:, f].values
    values_esrd = esrd.loc[:, f].values
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
            marker = dict(color=ctrl_color, line=dict(color='black',width=0.3), opacity=0.8),
            points='all',
            bandwidth = np.ptp(values_ctrl) / dist_num_bins,
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
            marker=dict(color=esrd_color, line=dict(color='black',width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(values_esrd) / dist_num_bins,
            opacity=0.8
        )
    )
    add_layout(fig, "", f"{f}", f"p-value: {result_df.at[f, 'pval_mv_fdr_bh']:0.2e}")
    fig.update_layout({'colorway': ['lime', 'fuchsia']})
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
    fig.update_layout(legend_y=1.01)
    fig.add_annotation(dict(font=dict(color='black', size=45),
                            x=-0.18,
                            y=1.25,
                            showarrow=False,
                            text=f"({string.ascii_lowercase[f_id]})",
                            textangle=0,
                            yanchor='top',
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    save_figure(fig, f"{path_save}/SupplementaryFigure2/{string.ascii_lowercase[f_id]}_{f}")

# Figure 2 =============================================================================================================
result_dict = {
    'feature': ['DNAmAgeHannumAA', 'DNAmAgeAA', 'IEAA', 'EEAA', 'DNAmPhenoAgeAA', 'DNAmGrimAgeAA'],
    'name': ['DNAmAgeHannum acceleration', 'DNAmAge acceleration', 'IEAA', 'EEAA', 'DNAmPhenoAge acceleration', 'DNAmGrimAge acceleration']
}
result_dict['pval_mv'] = np.zeros(len(result_dict['feature']))
for f_id, f in enumerate(result_dict['feature']):
    values_ctrl = ctrl.loc[:, f].values
    values_esrd = esrd.loc[:, f].values
    stat, pval = mannwhitneyu(values_ctrl, values_esrd, alternative='two-sided')
    result_dict['pval_mv'][f_id] = pval

result_dict = correct_pvalues(result_dict, ['pval_mv'])
result_df = pd.DataFrame(result_dict)
result_df.set_index('feature', inplace=True)
result_df.to_excel(f"{path_save}/Figure2/result.xlsx", index=True)

for f_id, f in enumerate(result_dict['feature']):
    values_ctrl = ctrl.loc[:, f].values
    values_esrd = esrd.loc[:, f].values
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
            marker = dict(color=ctrl_color, line=dict(color='black',width=0.3), opacity=0.8),
            points='all',
            bandwidth = np.ptp(values_ctrl) / dist_num_bins,
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
            marker=dict(color=esrd_color, line=dict(color='black',width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(values_esrd) / dist_num_bins,
            opacity=0.8
        )
    )
    add_layout(fig, "", f"{result_dict['name'][f_id]}", f"p-value: {result_df.at[f, 'pval_mv_fdr_bh']:0.2e}")
    fig.update_layout({'colorway': ['lime', 'fuchsia']})
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
    fig.update_layout(legend_y=1.01)
    fig.add_annotation(dict(font=dict(color='black', size=45),
                            x=-0.18,
                            y=1.25,
                            showarrow=False,
                            text=f"({string.ascii_lowercase[f_id]})",
                            textangle=0,
                            yanchor='top',
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    save_figure(fig, f"{path_save}/Figure2/{string.ascii_lowercase[f_id]}_{f}")

# Figure 3 =============================================================================================================
result_dict = {
    'feature': ['PhenoAgeAA',
                'White_blood_cell_count_(10^9/L)',
                'Lymphocyte_percent_(%)',
                'Creatinine_(umol/L)',
                'Mean_cell_volume_(fL)',
                'Albumin_(g/L)',
                'Alkaline_phosphatase_(U/L)',
                'Red_cell_distribution_width_(%)',
                'Glucose._serum_(mmol/L)_',
                'Log_C-reactive_protein_(mg/L)',
                ],
    'name': ['Phenotypic Age Acceleration',
             'White blood cell count (10^9/L)',
             'Lymphocyte percent (%)',
             'Creatinine (umol/L)',
             'Mean cell volume (fL)',
             'Albumin (g/L)',
             'Alkaline phosphatase (U/L)',
             'Red cell distribution width (%)',
             'Glucose. serum (mmol/L)',
             'Log C-reactive protein (mg/L)',
             ]
}
result_dict['pval_mv'] = np.zeros(len(result_dict['feature']))
for f_id, f in enumerate(result_dict['feature']):
    values_ctrl = ctrl.loc[:, f].values
    values_esrd = esrd.loc[:, f].values
    stat, pval = mannwhitneyu(values_ctrl, values_esrd, alternative='two-sided')
    result_dict['pval_mv'][f_id] = pval

result_dict = correct_pvalues(result_dict, ['pval_mv'])
result_df = pd.DataFrame(result_dict)
result_df.set_index('feature', inplace=True)
result_df.to_excel(f"{path_save}/Figure3/result.xlsx", index=True)

for f_id, f in enumerate(result_dict['feature']):
    values_ctrl = ctrl.loc[:, f].values
    values_esrd = esrd.loc[:, f].values
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
            marker = dict(color=ctrl_color, line=dict(color='black',width=0.3), opacity=0.8),
            points='all',
            bandwidth = np.ptp(values_ctrl) / dist_num_bins,
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
            marker=dict(color=esrd_color, line=dict(color='black',width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(values_esrd) / dist_num_bins,
            opacity=0.8
        )
    )
    if f == 'PhenoAgeAA':
        add_layout(fig, "", f"{result_dict['name'][f_id]}", f"p-value: {result_df.at[f, 'pval_mv']:0.2e}")
    else:
        add_layout(fig, "", f"{result_dict['name'][f_id]}", f"p-value: {result_df.at[f, 'pval_mv_fdr_bh']:0.2e}")
    fig.update_layout({'colorway': ['lime', 'fuchsia']})
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
    fig.update_layout(legend_y=1.01)
    fig.add_annotation(dict(font=dict(color='black', size=45),
                            x=-0.18,
                            y=1.25,
                            showarrow=False,
                            text=f"({string.ascii_lowercase[f_id]})",
                            textangle=0,
                            yanchor='top',
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    save_figure(fig, f"{path_save}/Figure3/{string.ascii_lowercase[f_id]}")

# Figure 4 =============================================================================================================
ctrl_test = pd.read_excel(f"{path}/{platform}/{dataset}/special/011_immuno_part3_and_part4_check_clocks/part3_part4_filtered_with_age_sex_16.xlsx", index_col='ID')

rmse = np.sqrt(mean_squared_error(ctrl_test.loc[:, 'Age'].values, ctrl_test.loc[:, 'ImmunoAge'].values))
mae = mean_absolute_error(ctrl_test.loc[:, 'Age'].values, ctrl_test.loc[:, 'ImmunoAge'].values)
print(f"RMSE in test controls: {rmse}")
print(f"MAE in test controls: {mae}")

values_ctrl = ctrl.loc[:, 'ImmunoAgeAA'].values
values_ctrl_test = ctrl_test.loc[:, 'ImmunoAgeAcc'].values
values_esrd = esrd.loc[:, 'ImmunoAgeAA'].values

stat_01, pval_01 = mannwhitneyu(values_ctrl, values_ctrl_test, alternative='two-sided')
stat_02, pval_02 = mannwhitneyu(values_ctrl, values_esrd, alternative='two-sided')
stat_12, pval_12 = mannwhitneyu(values_ctrl_test, values_esrd, alternative='two-sided')

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
save_figure(fig, f"{path_save}/Figure4/b")

formula = f"ImmunoAge ~ Age"
model_linear = smf.ols(formula=formula, data=ctrl).fit()
fig = go.Figure()
add_scatter_trace(fig, ctrl.loc[:, 'Age'].values, ctrl.loc[:, 'ImmunoAge'].values, f"Control")
add_scatter_trace(fig, ctrl.loc[:, 'Age'].values, model_linear.fittedvalues.values, "", "lines")
add_scatter_trace(fig, ctrl_test.loc[:, 'Age'].values, ctrl_test.loc[:, 'ImmunoAge'].values, f"Control (test)")
add_scatter_trace(fig, esrd.loc[:, 'Age'].values, esrd.loc[:, 'ImmunoAge'].values, f"ESRD")
add_layout(fig, f"Age", 'ipAGE', f"")
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
save_figure(fig, f"{path_save}/Figure4/a")

# Figure 5 a,b =========================================================================================================
result_dict = {
    'feature': ['Age', 'DNAmAgeHannum', 'DNAmAge', 'DNAmPhenoAge', 'DNAmGrimAge', 'PhenoAge', 'ImmunoAge'],
    'name': ['Age', 'DNAmAgeHannum', 'DNAmAge', 'DNAmPhenoAge', 'DNAmGrimAge', 'PhenotypicAge', 'ipAGE']
}

corr_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])
pval_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])
corr_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])
pval_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])

for f_id_1, f_1 in enumerate(result_dict['feature']):
    for f_id_2, f_2 in enumerate(result_dict['feature']):
        values_1_ctrl = ctrl.loc[:, f_1].values
        values_2_ctrl = ctrl.loc[:, f_2].values
        values_1_esrd = esrd.loc[:, f_1].values
        values_2_esrd = esrd.loc[:, f_2].values

        corr_ctrl, pval_ctrl = stats.pearsonr(values_1_ctrl, values_2_ctrl)
        corr_df_ctrl.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = corr_ctrl
        pval_df_ctrl.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = pval_ctrl
        corr_esrd, pval_esrd = stats.pearsonr(values_1_esrd, values_2_esrd)
        corr_df_esrd.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = corr_esrd
        pval_df_esrd.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = pval_esrd

selection = np.tri(pval_df_ctrl.shape[0], pval_df_ctrl.shape[1], -1, dtype=np.bool)

pval_df_ctrl_fdr = pval_df_ctrl.where(selection).stack().reset_index()
pval_df_ctrl_fdr.columns = ['row', 'col', 'pval']
_, pvals_corr, _, _ = multipletests(pval_df_ctrl_fdr.loc[:, 'pval'].values, 0.05, method='fdr_bh')
pval_df_ctrl_fdr['pval_fdr_bh'] = pvals_corr

pval_df_esrd_fdr = pval_df_esrd.where(selection).stack().reset_index()
pval_df_esrd_fdr.columns = ['row', 'col', 'pval']
_, pvals_corr, _, _ = multipletests(pval_df_esrd_fdr.loc[:, 'pval'].values, 0.05, method='fdr_bh')
pval_df_esrd_fdr['pval_fdr_bh'] = pvals_corr

for line_id in range(pval_df_ctrl_fdr.shape[0]):
    pval_df_ctrl.loc[pval_df_ctrl_fdr.at[line_id, 'row'], pval_df_ctrl_fdr.at[line_id, 'col']] = pval_df_ctrl_fdr.at[line_id, 'pval_fdr_bh']
    pval_df_ctrl.loc[pval_df_ctrl_fdr.at[line_id, 'col'], pval_df_ctrl_fdr.at[line_id, 'row']] = pval_df_ctrl_fdr.at[line_id, 'pval_fdr_bh']
    pval_df_esrd.loc[pval_df_esrd_fdr.at[line_id, 'row'], pval_df_esrd_fdr.at[line_id, 'col']] = pval_df_esrd_fdr.at[line_id, 'pval_fdr_bh']
    pval_df_esrd.loc[pval_df_esrd_fdr.at[line_id, 'col'], pval_df_esrd_fdr.at[line_id, 'row']] = pval_df_esrd_fdr.at[line_id, 'pval_fdr_bh']

for f_id_1, f_1 in enumerate(result_dict['feature']):
    pval_df_ctrl.loc[result_dict['name'][f_id_1], :] = -np.log10(pval_df_ctrl.loc[result_dict['name'][f_id_1], :])
    pval_df_esrd.loc[result_dict['name'][f_id_1], :] = -np.log10(pval_df_esrd.loc[result_dict['name'][f_id_1], :])

corr_df_ctrl = corr_df_ctrl.iloc[::-1]
mtx_to_plot = corr_df_ctrl.to_numpy()
cmap = plt.get_cmap("bwr").copy()
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-1, vmax=1)
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.046, pad=0.04)
cbar.set_label(r"$\mathrm{Correlation}$", horizontalalignment='center', fontsize=15)
ax.set_aspect("equal")
ax.set_xticks(np.arange(corr_df_ctrl.shape[0]))
ax.set_yticks(np.arange(corr_df_ctrl.shape[0]))
ax.set_xticklabels(corr_df_ctrl.columns.values)
ax.set_yticklabels(corr_df_ctrl.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
textcolors = ("black", "white")
for i in range(corr_df_ctrl.shape[0]):
    for j in range(corr_df_ctrl.shape[0]):
        color = 'black'
        text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=7)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure5/a_1.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure5/a_1.pdf", bbox_inches='tight', dpi=400)
plt.clf()

pval_df_ctrl = pval_df_ctrl.iloc[::-1]
mtx_to_plot = pval_df_ctrl.to_numpy()
cmap = plt.get_cmap("Reds").copy()
cmap.set_under('lightseagreen')
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.046, pad=0.04)
cbar.set_label(r"$-\log_{10}(\mathrm{p-value})$", horizontalalignment='center', fontsize=15)
ax.set_aspect("equal")
ax.set_xticks(np.arange(pval_df_ctrl.shape[0]))
ax.set_yticks(np.arange(pval_df_ctrl.shape[0]))
ax.set_xticklabels(pval_df_ctrl.columns.values)
ax.set_yticklabels(pval_df_ctrl.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
textcolors = ("black", "white")
for i in range(pval_df_ctrl.shape[0]):
    for j in range(pval_df_ctrl.shape[0]):
        color = textcolors[int(im.norm(data[i, j]) > threshold)]
        if np.isinf(mtx_to_plot[i, j]):
            text = ax.text(j, i, f"", ha="center", va="center", color=color, fontsize=7)
        else:
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=7)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure5/a_2.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure5/a_2.pdf", bbox_inches='tight', dpi=400)
plt.clf()

corr_df_esrd = corr_df_esrd.iloc[::-1]
mtx_to_plot = corr_df_esrd.to_numpy()
cmap = plt.get_cmap("bwr").copy()
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-1, vmax=1)
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.046, pad=0.04)
cbar.set_label(r"$\mathrm{Correlation}$", horizontalalignment='center', fontsize=15)
ax.set_aspect("equal")
ax.set_xticks(np.arange(corr_df_esrd.shape[0]))
ax.set_yticks(np.arange(corr_df_esrd.shape[0]))
ax.set_xticklabels(corr_df_esrd.columns.values)
ax.set_yticklabels(corr_df_esrd.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
textcolors = ("black", "white")
for i in range(corr_df_esrd.shape[0]):
    for j in range(corr_df_esrd.shape[0]):
        color = 'black'
        text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=7)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure5/b_1.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure5/b_1.pdf", bbox_inches='tight', dpi=400)
plt.clf()

pval_df_esrd = pval_df_esrd.iloc[::-1]
mtx_to_plot = pval_df_esrd.to_numpy()
cmap = plt.get_cmap("Reds").copy()
cmap.set_under('lightseagreen')
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.046, pad=0.04)
cbar.set_label(r"$-\log_{10}(\mathrm{p-value})$", horizontalalignment='center', fontsize=15)
ax.set_aspect("equal")
ax.set_xticks(np.arange(pval_df_esrd.shape[0]))
ax.set_yticks(np.arange(pval_df_esrd.shape[0]))
ax.set_xticklabels(pval_df_esrd.columns.values)
ax.set_yticklabels(pval_df_esrd.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
textcolors = ("black", "white")
for i in range(pval_df_esrd.shape[0]):
    for j in range(pval_df_esrd.shape[0]):
        color = textcolors[int(im.norm(data[i, j]) > threshold)]
        if np.isinf(mtx_to_plot[i, j]):
            text = ax.text(j, i, f"", ha="center", va="center", color=color, fontsize=7)
        else:
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=7)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure5/b_2.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure5/b_2.pdf", bbox_inches='tight', dpi=400)
plt.clf()

# Figure 5 c,d =========================================================================================================
result_dict = {
    'feature': ['DNAmAgeHannumAA', 'DNAmAgeAA', 'IEAA', 'EEAA', 'DNAmPhenoAgeAA', 'DNAmGrimAgeAA', 'PhenoAgeAA', 'ImmunoAgeAA'],
    'name': ['DNAmAgeHannumAcc', 'DNAmAgeAcc', 'IEAA', 'EEAA', 'DNAmPhenoAgeAcc', 'DNAmGrimAgeAcc', 'PhenotypicAgeAcc', 'ipAGEAcc']
}

corr_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])
pval_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])
corr_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])
pval_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(result_dict['feature']), len(result_dict['feature']))), index=result_dict['name'], columns=result_dict['name'])

for f_id_1, f_1 in enumerate(result_dict['feature']):
    for f_id_2, f_2 in enumerate(result_dict['feature']):
        values_1_ctrl = ctrl.loc[:, f_1].values
        values_2_ctrl = ctrl.loc[:, f_2].values
        values_1_esrd = esrd.loc[:, f_1].values
        values_2_esrd = esrd.loc[:, f_2].values

        corr_ctrl, pval_ctrl = stats.pearsonr(values_1_ctrl, values_2_ctrl)
        corr_df_ctrl.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = corr_ctrl
        pval_df_ctrl.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = pval_ctrl
        corr_esrd, pval_esrd = stats.pearsonr(values_1_esrd, values_2_esrd)
        corr_df_esrd.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = corr_esrd
        pval_df_esrd.loc[result_dict['name'][f_id_1], result_dict['name'][f_id_2]] = pval_esrd

selection = np.tri(pval_df_ctrl.shape[0], pval_df_ctrl.shape[1], -1, dtype=np.bool)

pval_df_ctrl_fdr = pval_df_ctrl.where(selection).stack().reset_index()
pval_df_ctrl_fdr.columns = ['row', 'col', 'pval']
_, pvals_corr, _, _ = multipletests(pval_df_ctrl_fdr.loc[:, 'pval'].values, 0.05, method='fdr_bh')
pval_df_ctrl_fdr['pval_fdr_bh'] = pvals_corr

pval_df_esrd_fdr = pval_df_esrd.where(selection).stack().reset_index()
pval_df_esrd_fdr.columns = ['row', 'col', 'pval']
_, pvals_corr, _, _ = multipletests(pval_df_esrd_fdr.loc[:, 'pval'].values, 0.05, method='fdr_bh')
pval_df_esrd_fdr['pval_fdr_bh'] = pvals_corr

for line_id in range(pval_df_ctrl_fdr.shape[0]):
    pval_df_ctrl.loc[pval_df_ctrl_fdr.at[line_id, 'row'], pval_df_ctrl_fdr.at[line_id, 'col']] = pval_df_ctrl_fdr.at[line_id, 'pval_fdr_bh']
    pval_df_ctrl.loc[pval_df_ctrl_fdr.at[line_id, 'col'], pval_df_ctrl_fdr.at[line_id, 'row']] = pval_df_ctrl_fdr.at[line_id, 'pval_fdr_bh']
    pval_df_esrd.loc[pval_df_esrd_fdr.at[line_id, 'row'], pval_df_esrd_fdr.at[line_id, 'col']] = pval_df_esrd_fdr.at[line_id, 'pval_fdr_bh']
    pval_df_esrd.loc[pval_df_esrd_fdr.at[line_id, 'col'], pval_df_esrd_fdr.at[line_id, 'row']] = pval_df_esrd_fdr.at[line_id, 'pval_fdr_bh']

for f_id_1, f_1 in enumerate(result_dict['feature']):
    pval_df_ctrl.loc[result_dict['name'][f_id_1], :] = -np.log10(pval_df_ctrl.loc[result_dict['name'][f_id_1], :])
    pval_df_esrd.loc[result_dict['name'][f_id_1], :] = -np.log10(pval_df_esrd.loc[result_dict['name'][f_id_1], :])

corr_df_ctrl = corr_df_ctrl.iloc[::-1]
mtx_to_plot = corr_df_ctrl.to_numpy()
cmap = plt.get_cmap("bwr").copy()
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-1, vmax=1)
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.046, pad=0.04)
cbar.set_label(r"$\mathrm{Correlation}$", horizontalalignment='center', fontsize=15)
ax.set_aspect("equal")
ax.set_xticks(np.arange(corr_df_ctrl.shape[0]))
ax.set_yticks(np.arange(corr_df_ctrl.shape[0]))
ax.set_xticklabels(corr_df_ctrl.columns.values)
ax.set_yticklabels(corr_df_ctrl.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
textcolors = ("black", "white")
for i in range(corr_df_ctrl.shape[0]):
    for j in range(corr_df_ctrl.shape[0]):
        color = 'black'
        text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=6)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure5/c_1.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure5/c_1.pdf", bbox_inches='tight', dpi=400)
plt.clf()

pval_df_ctrl = pval_df_ctrl.iloc[::-1]
mtx_to_plot = pval_df_ctrl.to_numpy()
cmap = plt.get_cmap("Reds").copy()
cmap.set_under('lightseagreen')
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.046, pad=0.04)
cbar.set_label(r"$-\log_{10}(\mathrm{p-value})$", horizontalalignment='center', fontsize=15)
ax.set_aspect("equal")
ax.set_xticks(np.arange(pval_df_ctrl.shape[0]))
ax.set_yticks(np.arange(pval_df_ctrl.shape[0]))
ax.set_xticklabels(pval_df_ctrl.columns.values)
ax.set_yticklabels(pval_df_ctrl.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
textcolors = ("black", "white")
for i in range(pval_df_ctrl.shape[0]):
    for j in range(pval_df_ctrl.shape[0]):
        color = textcolors[int(im.norm(data[i, j]) > threshold)]
        if np.isinf(mtx_to_plot[i, j]):
            text = ax.text(j, i, f"", ha="center", va="center", color=color, fontsize=7)
        else:
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=6)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure5/c_2.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure5/c_2.pdf", bbox_inches='tight', dpi=400)
plt.clf()

corr_df_esrd = corr_df_esrd.iloc[::-1]
mtx_to_plot = corr_df_esrd.to_numpy()
cmap = plt.get_cmap("bwr").copy()
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-1, vmax=1)
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.046, pad=0.04)
cbar.set_label(r"$\mathrm{Correlation}$", horizontalalignment='center', fontsize=15)
ax.set_aspect("equal")
ax.set_xticks(np.arange(corr_df_esrd.shape[0]))
ax.set_yticks(np.arange(corr_df_esrd.shape[0]))
ax.set_xticklabels(corr_df_esrd.columns.values)
ax.set_yticklabels(corr_df_esrd.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
textcolors = ("black", "white")
for i in range(corr_df_esrd.shape[0]):
    for j in range(corr_df_esrd.shape[0]):
        color = 'black'
        text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=6)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure5/d_1.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure5/d_1.pdf", bbox_inches='tight', dpi=400)
plt.clf()

pval_df_esrd = pval_df_esrd.iloc[::-1]
mtx_to_plot = pval_df_esrd.to_numpy()
cmap = plt.get_cmap("Reds").copy()
cmap.set_under('lightseagreen')
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.046, pad=0.04)
cbar.set_label(r"$-\log_{10}(\mathrm{p-value})$", horizontalalignment='center', fontsize=15)
ax.set_aspect("equal")
ax.set_xticks(np.arange(pval_df_esrd.shape[0]))
ax.set_yticks(np.arange(pval_df_esrd.shape[0]))
ax.set_xticklabels(pval_df_esrd.columns.values)
ax.set_yticklabels(pval_df_esrd.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
textcolors = ("black", "white")
for i in range(pval_df_esrd.shape[0]):
    for j in range(pval_df_esrd.shape[0]):
        color = textcolors[int(im.norm(data[i, j]) > threshold)]
        if np.isinf(mtx_to_plot[i, j]):
            text = ax.text(j, i, f"", ha="center", va="center", color=color, fontsize=7)
        else:
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=6)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure5/d_2.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure5/d_2.pdf", bbox_inches='tight', dpi=400)
plt.clf()

# Figure 6 =============================================================================================================
# SupplementaryTable9 ==================================================================================================
with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    immuno_features = f.read().splitlines()
result_dict = {'feature': immuno_features}
result_dict['pval_mv'] = np.zeros(len(result_dict['feature']))
for f_id, f in enumerate(result_dict['feature']):
    values_ctrl = ctrl.loc[:, f].values
    values_esrd = esrd.loc[:, f].values
    stat, pval = mannwhitneyu(values_ctrl, values_esrd, alternative='two-sided')
    result_dict['pval_mv'][f_id] = pval

result_dict = correct_pvalues(result_dict, ['pval_mv'])
result_df = pd.DataFrame(result_dict)
venn_lists = {'ESRD': result_df.loc[result_df['pval_mv_fdr_bh'] < 0.05, 'feature'].tolist()}
result_df.set_index('feature', inplace=True)
result_df.sort_values(['pval_mv'], ascending=[True], inplace=True)
result_df.rename(
    columns={'pval_mv': 'Mann–Whitney U test p-value', 'pval_mv_fdr_bh': 'Mann–Whitney U test p-value (FDR)'},
    inplace=True
)
result_df.drop('pval_mv_bonferroni', axis=1, inplace=True)
result_df.to_excel(f"{path_save}/SupplementaryTable9/Disease.xlsx", index=True)
top_features = ['CSF1', 'CXCL9']
top_features_ranges = {'CSF1': [-100, 2500], 'CXCL9': [-1000, 20000]}
top_features_bandwidth={'CSF1': {'Control': 50, 'ESRD': 90}, 'CXCL9': {'Control': 800, 'ESRD': 800}}

for f_id, f in enumerate(top_features):
    values_ctrl = ctrl.loc[:, f].values
    values_esrd = esrd.loc[:, f].values
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
            marker = dict(color=ctrl_color, line=dict(color='black',width=0.3), opacity=0.8),
            points='all',
            bandwidth=top_features_bandwidth[f]['Control'],
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
            marker=dict(color=esrd_color, line=dict(color='black',width=0.3), opacity=0.8),
            points='all',
            bandwidth=top_features_bandwidth[f]['ESRD'],
            opacity=0.8
        )
    )
    add_layout(fig, "", f"{f}", f"p-value: {result_df.at[f, 'Mann–Whitney U test p-value (FDR)']:0.2e}")
    fig.update_layout({'colorway': ['lime', 'fuchsia']})
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_yaxes(autorange=False)
    fig.update_layout(yaxis_range=top_features_ranges[f])
    fig.update_layout(
        margin=go.layout.Margin(
            l=130,
            r=20,
            b=50,
            t=90,
            pad=0
        )
    )
    fig.update_layout(legend_y=1.01)
    fig.add_annotation(dict(font=dict(color='black', size=45),
                            x=-0.18,
                            y=1.25,
                            showarrow=False,
                            text=f"({string.ascii_lowercase[f_id+1]})",
                            textangle=0,
                            yanchor='top',
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    save_figure(fig, f"{path_save}/Figure6/{string.ascii_lowercase[f_id+1]}_{f}")

xs = -np.log10(result_df.loc[:, 'Mann–Whitney U test p-value (FDR)'].values)[::-1]
ys = result_df.index.values[::-1]

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=xs,
        y=list(range(len(xs))),
        orientation='h',
        marker=dict(color='red', opacity=0.9)
    )
)
fig.add_trace(
    go.Scatter(
        x=[-np.log10(0.05), -np.log10(0.05)],
        y=[-1, len(xs)],
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
        tickvals = list(range(len(xs))),
        ticktext = ys
    )
)
fig.update_yaxes(autorange=False)
fig.update_layout(yaxis_range=[-1, len(xs)])
fig.update_yaxes(tickfont_size=24)
fig.update_xaxes(tickfont_size=30)
fig.update_layout(
    autosize=False,
    width=800,
    height=1400,
    margin=go.layout.Margin(
        l=175,
        r=20,
        b=100,
        t=40,
        pad=0
    )
)
fig.add_annotation(dict(font=dict(color='black', size=65),
                        x=-0.29,
                        y=1.04,
                        showarrow=False,
                        text=f"(a)",
                        textangle=0,
                        yanchor='top',
                        xanchor='left',
                        xref="paper",
                        yref="paper"))
save_figure(fig, f"{path_save}/Figure6/a")

# Figure 7a ============================================================================================================
features_2 = ['Age', 'DNAmAgeHannum', 'DNAmAge', 'DNAmPhenoAge', 'DNAmGrimAge', 'PhenoAge', 'ImmunoAge']
names_2 = ['Age', 'DNAmAgeHannum', 'DNAmAge', 'DNAmPhenoAge', 'DNAmGrimAge', 'PhenotypicAge', 'ipAGE']
with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    features_1 = f.read().splitlines()
    names_1 = features_1.copy()

corr_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2))), index=names_1, columns=names_2)
pval_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2))), index=names_1, columns=names_2)

age_col_names = []
for n in names_2:
    age_col_names.append(f"{n} correlation")
    age_col_names.append(f"{n} p-value")
    age_col_names.append(f"{n} p-value (FDR)")
age_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2) * 3)), index=names_1, columns=age_col_names)
age_df_ctrl.index.name = 'feature'

for f_id_2, f_2 in enumerate(features_2):
    for f_id_1, f_1 in enumerate(features_1):
        values_1_ctrl = ctrl.loc[:, f_1].values
        values_2_ctrl = ctrl.loc[:, f_2].values
        corr_ctrl, pval_ctrl = stats.pearsonr(values_1_ctrl, values_2_ctrl)
        corr_df_ctrl.loc[names_1[f_id_1], names_2[f_id_2]] = corr_ctrl
        pval_df_ctrl.loc[names_1[f_id_1], names_2[f_id_2]] = pval_ctrl

    age_df_ctrl.loc[:, f"{names_2[f_id_2]} correlation"] = corr_df_ctrl.loc[:, names_2[f_id_2]]
    age_df_ctrl.loc[:, f"{names_2[f_id_2]} p-value"] = pval_df_ctrl.loc[:, names_2[f_id_2]]

for f_id_2, f_2 in enumerate(features_2):
    _, pvals_corr, _, _ = multipletests(pval_df_ctrl.loc[:, names_2[f_id_2]].values, 0.05, method='fdr_bh')
    age_df_ctrl.loc[:, f"{names_2[f_id_2]} p-value (FDR)"] = pvals_corr
    pval_df_ctrl.loc[:, names_2[f_id_2]] = -np.log10(pvals_corr)

venn_lists['Age_Control'] = age_df_ctrl.index[age_df_ctrl[f"Age p-value (FDR)"] < 0.05].tolist()

age_df_ctrl.to_excel(f"{path_save}/SupplementaryTable9/Age_Control.xlsx", index=True)

corr_df_ctrl = corr_df_ctrl.iloc[::-1]
mtx_to_plot = corr_df_ctrl.to_numpy()
cmap = plt.get_cmap("bwr").copy()
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-1, vmax=1)
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.05, pad=0.03, shrink=0.155)
cbar.set_label(r"$\mathrm{Correlation}$", horizontalalignment='center', fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(corr_df_ctrl.shape[1]))
ax.set_yticks(np.arange(corr_df_ctrl.shape[0]))
ax.set_xticklabels(corr_df_ctrl.columns.values)
ax.set_yticklabels(corr_df_ctrl.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='x', which='major', labelsize=6)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='y', which='major', labelsize=5)
ax.tick_params(axis='y', which='minor', labelsize=5)
textcolors = ("black", "white")
for i in range(corr_df_ctrl.shape[0]):
    for j in range(corr_df_ctrl.shape[1]):
        color = 'black'
        text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=3)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure7/a_1.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure7/a_1.pdf", bbox_inches='tight', dpi=400)
plt.clf()

pval_df_ctrl = pval_df_ctrl.iloc[::-1]
mtx_to_plot = pval_df_ctrl.to_numpy()
cmap = plt.get_cmap("Reds").copy()
cmap.set_under('lightseagreen')
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.05, pad=0.03, shrink=0.155)
cbar.set_label(r"$-\log_{10}(\mathrm{p-value})$", horizontalalignment='center', fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(pval_df_ctrl.shape[1]))
ax.set_yticks(np.arange(pval_df_ctrl.shape[0]))
ax.set_xticklabels(pval_df_ctrl.columns.values)
ax.set_yticklabels(pval_df_ctrl.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='x', which='major', labelsize=6)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='y', which='major', labelsize=5)
ax.tick_params(axis='y', which='minor', labelsize=5)
textcolors = ("black", "white")
for i in range(pval_df_ctrl.shape[0]):
    for j in range(pval_df_ctrl.shape[1]):
        color = textcolors[int(im.norm(data[i, j]) > threshold)]
        if np.isinf(mtx_to_plot[i, j]):
            text = ax.text(j, i, f"", ha="center", va="center", color=color, fontsize=3)
        else:
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=3)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure7/a_2.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure7/a_2.pdf", bbox_inches='tight', dpi=400)
plt.clf()

# Figure 7b ============================================================================================================
features_2 = ['Age', 'DNAmAgeHannum', 'DNAmAge', 'DNAmPhenoAge', 'DNAmGrimAge', 'PhenoAge', 'ImmunoAge', 'Dialysis_(months)']
names_2 = ['Age', 'DNAmAgeHannum', 'DNAmAge', 'DNAmPhenoAge', 'DNAmGrimAge', 'PhenotypicAge', 'ipAGE', 'Dialysis(months)']
with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    features_1 = f.read().splitlines()
    names_1 = features_1.copy()

features_1 += ['Dialysis_(months)']
names_1 += ['Dialysis(months)']

age_col_names = []
for n in names_2:
    age_col_names.append(f"{n} correlation")
    age_col_names.append(f"{n} p-value")
    age_col_names.append(f"{n} p-value (FDR)")
age_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2) * 3)), index=names_1, columns=age_col_names)
age_df_esrd.index.name = 'feature'

corr_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2))), index=names_1, columns=names_2)
pval_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2))), index=names_1, columns=names_2)

for f_id_2, f_2 in enumerate(features_2):
    for f_id_1, f_1 in enumerate(features_1):
        values_1_esrd = esrd.loc[:, f_1].values
        values_2_esrd = esrd.loc[:, f_2].values
        corr_esrd, pval_esrd = stats.pearsonr(values_1_esrd, values_2_esrd)
        corr_df_esrd.loc[names_1[f_id_1], names_2[f_id_2]] = corr_esrd
        pval_df_esrd.loc[names_1[f_id_1], names_2[f_id_2]] = pval_esrd

    age_df_esrd.loc[:, f"{names_2[f_id_2]} correlation"] = corr_df_esrd.loc[:, names_2[f_id_2]]
    age_df_esrd.loc[:, f"{names_2[f_id_2]} p-value"] = pval_df_esrd.loc[:, names_2[f_id_2]]

for f_id_2, f_2 in enumerate(features_2):
    _, pvals_corr, _, _ = multipletests(pval_df_esrd.loc[:, names_2[f_id_2]].values, 0.05, method='fdr_bh')
    age_df_esrd.loc[:, f"{names_2[f_id_2]} p-value (FDR)"] = pvals_corr
    pval_df_esrd.loc[:, names_2[f_id_2]] = -np.log10(pvals_corr)

age_df_esrd.to_excel(f"{path_save}/SupplementaryTable9/Age_ESRD.xlsx", index=True)

corr_df_esrd = corr_df_esrd.iloc[::-1]
mtx_to_plot = corr_df_esrd.to_numpy()
cmap = plt.get_cmap("bwr").copy()
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-1, vmax=1)
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.05, pad=0.03, shrink=0.175)
cbar.set_label(r"$\mathrm{Correlation}$", horizontalalignment='center', fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(corr_df_esrd.shape[1]))
ax.set_yticks(np.arange(corr_df_esrd.shape[0]))
ax.set_xticklabels(corr_df_esrd.columns.values)
ax.set_yticklabels(corr_df_esrd.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='x', which='major', labelsize=6)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='y', which='major', labelsize=5)
ax.tick_params(axis='y', which='minor', labelsize=5)
textcolors = ("black", "white")
for i in range(corr_df_esrd.shape[0]):
    for j in range(corr_df_esrd.shape[1]):
        color = 'black'
        text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=3)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure7/b_1.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure7/b_1.pdf", bbox_inches='tight', dpi=400)
plt.clf()

pval_df_esrd = pval_df_esrd.iloc[::-1]
mtx_to_plot = pval_df_esrd.to_numpy()
cmap = plt.get_cmap("Reds").copy()
cmap.set_under('lightseagreen')
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.05, pad=0.03, shrink=0.175)
cbar.set_label(r"$-\log_{10}(\mathrm{p-value})$", horizontalalignment='center', fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(pval_df_esrd.shape[1]))
ax.set_yticks(np.arange(pval_df_esrd.shape[0]))
ax.set_xticklabels(pval_df_esrd.columns.values)
ax.set_yticklabels(pval_df_esrd.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='x', which='major', labelsize=6)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='y', which='major', labelsize=5)
ax.tick_params(axis='y', which='minor', labelsize=5)
textcolors = ("black", "white")
for i in range(pval_df_esrd.shape[0]):
    for j in range(pval_df_esrd.shape[1]):
        color = textcolors[int(im.norm(data[i, j]) > threshold)]
        if np.isinf(mtx_to_plot[i, j]):
            text = ax.text(j, i, f"", ha="center", va="center", color=color, fontsize=3)
        else:
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=3)
fig.tight_layout()
plt.savefig(f"{path_save}/Figure7/b_2.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure7/b_2.pdf", bbox_inches='tight', dpi=400)
plt.clf()

# Figure 7 Scatters ====================================================================================================
top_features = ['CXCL9', 'VEGFA', 'CCL2']
top_features_ranges = {'CXCL9': [0, 5000], 'VEGFA': [0, 250], 'CCL2': [0, 500]}

for f_id, f in enumerate(top_features):
    formula = f"{f} ~ Age"
    model_linear = smf.ols(formula=formula, data=ctrl).fit()
    fig = go.Figure()
    add_scatter_trace(fig, ctrl.loc[:, 'Age'].values, ctrl.loc[:, f].values, f"Control")
    add_scatter_trace(fig, ctrl.loc[:, 'Age'].values, model_linear.fittedvalues.values, "", "lines")
    add_scatter_trace(fig, esrd.loc[:, 'Age'].values, esrd.loc[:, f].values, f"ESRD")
    add_layout(fig, f"Age", f'{f}', f"")
    fig.update_layout({'colorway': ['lime', 'lime', 'fuchsia']})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(
        margin=go.layout.Margin(
            l=120,
            r=20,
            b=80,
            t=65,
            pad=0
        )
    )
    fig.update_yaxes(autorange=False)
    fig.update_xaxes(autorange=False)
    fig.update_layout(yaxis_range=top_features_ranges[f])
    fig.update_layout(xaxis_range=[10, 100])
    fig.add_annotation(dict(font=dict(color='black', size=45),
                            x=-0.22,
                            y=1.20,
                            showarrow=False,
                            text=f"({string.ascii_lowercase[f_id+4]})",
                            textangle=0,
                            yanchor='top',
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    save_figure(fig, f"{path_save}/Figure7/{string.ascii_lowercase[f_id+4]}_{f}")

# Figure 7 Venn ========================================================================================================
venn_sets = [set(x) for x in venn_lists.values()]
venn_tags = [x for x in venn_lists.keys()]
venn_sections = get_sections(venn_sets)
labels = vennrout.get_labels(list(venn_lists.values()), fill=['number'])

fig, ax = plt.subplots()
venn = venn2(
    subsets=(set(venn_lists['ESRD']), set(venn_lists['Age_Control'])),
    set_labels = (' Associated\nwith ESRD', 'Associated with age \n  (in Control group)'),
    set_colors=('r', 'g'),
    alpha = 0.5)
venn2_circles(subsets=(set(venn_lists['ESRD']), set(venn_lists['Age_Control'])))
for text in venn.set_labels:
    text.set_fontsize(16)
for text in venn.subset_labels:
    text.set_fontsize(25)
plt.savefig(f"{path_save}/Figure7/c.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/Figure7/c.pdf", bbox_inches='tight', dpi=400)
plt.clf()

# Figure 7 Table =======================================================================================================
features = ['Age', 'DNAmAgeHannum', 'DNAmAge', 'DNAmPhenoAge', 'DNAmGrimAge', 'PhenotypicAge', 'ipAGE']
columns = ['Control group', 'ESRD group']

table7 = pd.DataFrame(data=np.zeros(shape=(len(features), len(columns))), index=features, columns=columns)
table7_lists = {}
for f in features:
    table7.loc[f, 'Control group'] = len(age_df_ctrl.index[age_df_ctrl[f"{f} p-value (FDR)"] < 0.05].tolist())
    table7.loc[f, 'ESRD group'] = len(age_df_esrd.index[age_df_esrd[f"{f} p-value (FDR)"] < 0.05].tolist())
    table7_lists[f] = age_df_ctrl.index[age_df_ctrl[f"{f} p-value (FDR)"] < 0.05].tolist()

fig = go.Figure()
fig.add_trace(
    go.Table(
        header=dict(values=['Type of age'] + columns,
                    fill_color='paleturquoise',
                    align='left',
                    font_size=25),
        cells=dict(values=[table7.index.values, table7['Control group'], table7['ESRD group']],
                   fill_color='lavender',
                   align='left',
                   font_size=25,
                   height=35),
        columnwidth = [70, 30, 30]
    )
)
fig.update_layout(
    autosize=False,
    width=600,
    height=430,
    margin=go.layout.Margin(
        l=70,
        r=20,
        b=20,
        t=65,
        pad=0
    )
)
fig.add_annotation(dict(font=dict(color='black', size=45),
                        x=-0.13,
                        y=1.20,
                        showarrow=False,
                        text=f"(d)",
                        textangle=0,
                        yanchor='top',
                        xanchor='left',
                        xref="paper",
                        yref="paper"))
save_figure(fig, f"{path_save}/Figure7/d_1")

table7_sets = [set(x) for x in table7_lists.values()]
table7_tags = [x for x in table7_lists.keys()]
table7_sections = get_sections(table7_sets)
upset_df = pd.DataFrame(index=list(age_df_ctrl.index.values))
for k, v in table7_lists.items():
    upset_df[k] = upset_df.index.isin(v)
upset_df = upset_df.set_index(list(table7_lists.keys()))
fig = upset.UpSet(upset_df, subset_size='count', show_counts=True, min_degree=1, sort_categories_by=None).plot()
plt.savefig(f"{path_save}/Figure7/d_2.png", bbox_inches='tight')
plt.savefig(f"{path_save}/Figure7/d_2.pdf", bbox_inches='tight')
plt.clf()

# Supplementary Figure 3a ==============================================================================================
features_2 = ['DNAmAgeHannumAA', 'DNAmAgeAA', 'IEAA', 'EEAA', 'DNAmPhenoAgeAA', 'DNAmGrimAgeAA', 'ImmunoAgeAA']
names_2 = ['DNAmAgeHannumAcc', 'DNAmAgeAcc', 'IEAA', 'EEAA', 'DNAmPhenoAgeAcc', 'DNAmGrimAgeAcc', 'ipAGEAcc']
with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    features_1 = f.read().splitlines()
    names_1 = features_1.copy()

corr_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2))), index=names_1, columns=names_2)
pval_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2))), index=names_1, columns=names_2)

age_col_names = []
for n in names_2:
    age_col_names.append(f"{n} correlation")
    age_col_names.append(f"{n} p-value")
    age_col_names.append(f"{n} p-value (FDR)")
age_acc_df_ctrl = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2) * 3)), index=names_1, columns=age_col_names)
age_acc_df_ctrl.index.name = 'feature'

for f_id_2, f_2 in enumerate(features_2):
    for f_id_1, f_1 in enumerate(features_1):
        values_1_ctrl = ctrl.loc[:, f_1].values
        values_2_ctrl = ctrl.loc[:, f_2].values
        corr_ctrl, pval_ctrl = stats.pearsonr(values_1_ctrl, values_2_ctrl)
        corr_df_ctrl.loc[names_1[f_id_1], names_2[f_id_2]] = corr_ctrl
        pval_df_ctrl.loc[names_1[f_id_1], names_2[f_id_2]] = pval_ctrl

    age_acc_df_ctrl.loc[:, f"{names_2[f_id_2]} correlation"] = corr_df_ctrl.loc[:, names_2[f_id_2]]
    age_acc_df_ctrl.loc[:, f"{names_2[f_id_2]} p-value"] = pval_df_ctrl.loc[:, names_2[f_id_2]]

for f_id_2, f_2 in enumerate(features_2):
    _, pvals_corr, _, _ = multipletests(pval_df_ctrl.loc[:, names_2[f_id_2]].values, 0.05, method='fdr_bh')
    age_acc_df_ctrl.loc[:, f"{names_2[f_id_2]} p-value (FDR)"] = pvals_corr
    pval_df_ctrl.loc[:, names_2[f_id_2]] = -np.log10(pvals_corr)

age_acc_df_ctrl.to_excel(f"{path_save}/SupplementaryTable9/Age_Acc_Control.xlsx", index=True)

corr_df_ctrl = corr_df_ctrl.iloc[::-1]
mtx_to_plot = corr_df_ctrl.to_numpy()
cmap = plt.get_cmap("bwr").copy()
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-1, vmax=1)
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.05, pad=0.03, shrink=0.15)
cbar.set_label(r"$\mathrm{Correlation}$", horizontalalignment='center', fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(corr_df_ctrl.shape[1]))
ax.set_yticks(np.arange(corr_df_ctrl.shape[0]))
ax.set_xticklabels(corr_df_ctrl.columns.values)
ax.set_yticklabels(corr_df_ctrl.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='x', which='major', labelsize=6)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='y', which='major', labelsize=5)
ax.tick_params(axis='y', which='minor', labelsize=5)
textcolors = ("black", "white")
for i in range(corr_df_ctrl.shape[0]):
    for j in range(corr_df_ctrl.shape[1]):
        color = 'black'
        text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=3)
fig.tight_layout()
plt.savefig(f"{path_save}/SupplementaryFigure3/a_1.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/SupplementaryFigure3/a_1.pdf", bbox_inches='tight', dpi=400)
plt.clf()

pval_df_ctrl = pval_df_ctrl.iloc[::-1]
mtx_to_plot = pval_df_ctrl.to_numpy()
cmap = plt.get_cmap("Reds").copy()
cmap.set_under('lightseagreen')
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.05, pad=0.03, shrink=0.15)
cbar.set_label(r"$-\log_{10}(\mathrm{p-value})$", horizontalalignment='center', fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(pval_df_ctrl.shape[1]))
ax.set_yticks(np.arange(pval_df_ctrl.shape[0]))
ax.set_xticklabels(pval_df_ctrl.columns.values)
ax.set_yticklabels(pval_df_ctrl.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='x', which='major', labelsize=6)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='y', which='major', labelsize=5)
ax.tick_params(axis='y', which='minor', labelsize=5)
textcolors = ("black", "white")
for i in range(pval_df_ctrl.shape[0]):
    for j in range(pval_df_ctrl.shape[1]):
        color = textcolors[int(im.norm(data[i, j]) > threshold)]
        if np.isinf(mtx_to_plot[i, j]):
            text = ax.text(j, i, f"", ha="center", va="center", color=color, fontsize=3)
        else:
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=3)
fig.tight_layout()
plt.savefig(f"{path_save}/SupplementaryFigure3/a_2.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/SupplementaryFigure3/a_2.pdf", bbox_inches='tight', dpi=400)
plt.clf()

# Supplementary Figure 3b ============================================================================================================
features_2 = ['DNAmAgeHannumAA', 'DNAmAgeAA', 'IEAA', 'EEAA', 'DNAmPhenoAgeAA', 'DNAmGrimAgeAA', 'ImmunoAgeAA']
names_2 = ['DNAmAgeHannumAcc', 'DNAmAgeAcc', 'IEAA', 'EEAA', 'DNAmPhenoAgeAcc', 'DNAmGrimAgeAcc', 'ipAGEAcc']
with open(f'{path}/{platform}/{dataset}/features/immuno.txt') as f:
    features_1 = f.read().splitlines()
    names_1 = features_1.copy()
features_1 += ['Dialysis_(months)']
names_1 += ['Dialysis(months)']

age_col_names = []
for n in names_2:
    age_col_names.append(f"{n} correlation")
    age_col_names.append(f"{n} p-value")
    age_col_names.append(f"{n} p-value (FDR)")
age_acc_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2) * 3)), index=names_1, columns=age_col_names)
age_acc_df_esrd.index.name = 'feature'

corr_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2))), index=names_1, columns=names_2)
pval_df_esrd = pd.DataFrame(data=np.zeros(shape=(len(names_1), len(names_2))), index=names_1, columns=names_2)

for f_id_2, f_2 in enumerate(features_2):
    for f_id_1, f_1 in enumerate(features_1):
        values_1_esrd = esrd.loc[:, f_1].values
        values_2_esrd = esrd.loc[:, f_2].values
        corr_esrd, pval_esrd = stats.pearsonr(values_1_esrd, values_2_esrd)
        corr_df_esrd.loc[names_1[f_id_1], names_2[f_id_2]] = corr_esrd
        pval_df_esrd.loc[names_1[f_id_1], names_2[f_id_2]] = pval_esrd

    age_acc_df_esrd.loc[:, f"{names_2[f_id_2]} correlation"] = corr_df_esrd.loc[:, names_2[f_id_2]]
    age_acc_df_esrd.loc[:, f"{names_2[f_id_2]} p-value"] = pval_df_esrd.loc[:, names_2[f_id_2]]

for f_id_2, f_2 in enumerate(features_2):
    _, pvals_corr, _, _ = multipletests(pval_df_esrd.loc[:, names_2[f_id_2]].values, 0.05, method='fdr_bh')
    age_acc_df_esrd.loc[:, f"{names_2[f_id_2]} p-value (FDR)"] = pvals_corr
    pval_df_esrd.loc[:, names_2[f_id_2]] = -np.log10(pvals_corr)

venn_lists['Age_Acc_ESRD'] = age_acc_df_esrd.index[age_acc_df_esrd[f"ipAGEAcc p-value (FDR)"] < 0.05].tolist()

age_acc_df_esrd.to_excel(f"{path_save}/SupplementaryTable9/Age_Acc_ESRD.xlsx", index=True)

corr_df_esrd = corr_df_esrd.iloc[::-1]
mtx_to_plot = corr_df_esrd.to_numpy()
cmap = plt.get_cmap("bwr").copy()
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-1, vmax=1)
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.05, pad=0.03, shrink=0.15)
cbar.set_label(r"$\mathrm{Correlation}$", horizontalalignment='center', fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(corr_df_esrd.shape[1]))
ax.set_yticks(np.arange(corr_df_esrd.shape[0]))
ax.set_xticklabels(corr_df_esrd.columns.values)
ax.set_yticklabels(corr_df_esrd.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='x', which='major', labelsize=6)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='y', which='major', labelsize=5)
ax.tick_params(axis='y', which='minor', labelsize=5)
textcolors = ("black", "white")
for i in range(corr_df_esrd.shape[0]):
    for j in range(corr_df_esrd.shape[1]):
        color = 'black'
        text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=3)
fig.tight_layout()
plt.savefig(f"{path_save}/SupplementaryFigure3/b_1.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/SupplementaryFigure3/b_1.pdf", bbox_inches='tight', dpi=400)
plt.clf()

pval_df_esrd = pval_df_esrd.iloc[::-1]
mtx_to_plot = pval_df_esrd.to_numpy()
cmap = plt.get_cmap("Reds").copy()
cmap.set_under('lightseagreen')
fig, ax = plt.subplots()
im = ax.imshow(mtx_to_plot, cmap=cmap, vmin=-np.log10(0.05))
cbar = ax.figure.colorbar(im, ax=ax, location='top', fraction=0.05, pad=0.03, shrink=0.15)
cbar.set_label(r"$-\log_{10}(\mathrm{p-value})$", horizontalalignment='center', fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax.set_aspect(0.5)
ax.set_xticks(np.arange(pval_df_esrd.shape[1]))
ax.set_yticks(np.arange(pval_df_esrd.shape[0]))
ax.set_xticklabels(pval_df_esrd.columns.values)
ax.set_yticklabels(pval_df_esrd.index.values)
plt.setp(ax.get_xticklabels(), rotation=90)
data = im.get_array()
threshold = im.norm(data.max()) / 2.
ax.tick_params(axis='x', which='major', labelsize=6)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='y', which='major', labelsize=5)
ax.tick_params(axis='y', which='minor', labelsize=5)
textcolors = ("black", "white")
for i in range(pval_df_esrd.shape[0]):
    for j in range(pval_df_esrd.shape[1]):
        color = textcolors[int(im.norm(data[i, j]) > threshold)]
        if np.isinf(mtx_to_plot[i, j]):
            text = ax.text(j, i, f"", ha="center", va="center", color=color, fontsize=3)
        else:
            text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=3)
fig.tight_layout()
plt.savefig(f"{path_save}/SupplementaryFigure3/b_2.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/SupplementaryFigure3/b_2.pdf", bbox_inches='tight', dpi=400)
plt.clf()

# Supplementary Figure 3 Scatters ======================================================================================
top_features = ['CXCL9']
top_features_ranges = {'CXCL9': [0, 15000]}

for f_id, f in enumerate(top_features):
    formula = f"{f} ~ ImmunoAgeAA"
    model_linear = smf.ols(formula=formula, data=esrd).fit()
    fig = go.Figure()
    add_scatter_trace(fig, ctrl.loc[:, 'ImmunoAgeAA'].values, ctrl.loc[:, f].values, f"Control")
    add_scatter_trace(fig, esrd.loc[:, 'ImmunoAgeAA'].values, model_linear.fittedvalues.values, "", "lines")
    add_scatter_trace(fig, esrd.loc[:, 'ImmunoAgeAA'].values, esrd.loc[:, f].values, f"ESRD")
    add_layout(fig, f"ipAGE acceleration", f'{f}', f"")
    fig.update_layout({'colorway': ['lime', 'fuchsia', 'fuchsia']})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(
        margin=go.layout.Margin(
            l=140,
            r=20,
            b=80,
            t=65,
            pad=0
        )
    )
    fig.update_yaxes(autorange=False)
    fig.update_xaxes(autorange=False)
    fig.update_layout(yaxis_range=top_features_ranges[f])
    fig.update_layout(xaxis_range=[-50, 200])
    fig.add_annotation(dict(font=dict(color='black', size=45),
                            x=-0.26,
                            y=1.20,
                            showarrow=False,
                            text=f"({string.ascii_lowercase[f_id+4]})",
                            textangle=0,
                            yanchor='top',
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    save_figure(fig, f"{path_save}/SupplementaryFigure3/{string.ascii_lowercase[f_id+4]}_{f}")

# Supplementary Figure 3 Venn ==========================================================================================
venn_sets = [set(x) for x in venn_lists.values()]
venn_tags = [x for x in venn_lists.keys()]
venn_sections = get_sections(venn_sets)
labels = vennrout.get_labels(list(venn_lists.values()), fill=['number'])

fig, ax = plt.subplots()
venn = venn3(
    subsets=(set(venn_lists['ESRD']), set(venn_lists['Age_Control']), set(venn_lists['Age_Acc_ESRD'])),
    set_labels = (' Associated\nwith ESRD', 'Associated with age \n  (in Control group)', 'Associated with \n ipAGE acceleration \n(in ESRD group)'),
    set_colors=('r', 'g', 'b'),
    alpha = 0.5)
venn3_circles(subsets=(set(venn_lists['ESRD']), set(venn_lists['Age_Control']), set(venn_lists['Age_Acc_ESRD'])))
for text in venn.set_labels:
    text.set_fontsize(16)
for text in venn.subset_labels:
    text.set_fontsize(18)
plt.savefig(f"{path_save}/SupplementaryFigure3/c.png", bbox_inches='tight', dpi=400)
plt.savefig(f"{path_save}/SupplementaryFigure3/c.pdf", bbox_inches='tight', dpi=400)
plt.clf()

# Supplementary Figure 3 Table =========================================================================================
features = ['DNAmAgeHannumAcc', 'DNAmAgeAcc', 'IEAA', 'EEAA', 'DNAmPhenoAgeAcc', 'DNAmGrimAgeAcc', 'ipAGEAcc']
columns = ['Control group', 'ESRD group']

table7 = pd.DataFrame(data=np.zeros(shape=(len(features), len(columns))), index=features, columns=columns)
table7_lists = {}
for f in features:
    table7.loc[f, 'Control group'] = len(age_acc_df_ctrl.index[age_acc_df_ctrl[f"{f} p-value (FDR)"] < 0.05].tolist())
    table7.loc[f, 'ESRD group'] = len(age_acc_df_esrd.index[age_acc_df_esrd[f"{f} p-value (FDR)"] < 0.05].tolist())
    table7_lists[f] = age_acc_df_ctrl.index[age_acc_df_ctrl[f"{f} p-value (FDR)"] < 0.05].tolist()
fig = go.Figure()
fig.add_trace(
    go.Table(
        header=dict(values=['Type of age'] + columns,
                    fill_color='paleturquoise',
                    align='left',
                    font_size=25),
        cells=dict(values=[table7.index.values, table7['Control group'], table7['ESRD group']],
                   fill_color='lavender',
                   align='left',
                   font_size=25,
                   height=35),
        columnwidth = [80, 30, 30]
    )
)
fig.update_layout(
    autosize=False,
    width=600,
    height=430,
    margin=go.layout.Margin(
        l=70,
        r=20,
        b=20,
        t=65,
        pad=0
    )
)
fig.add_annotation(dict(font=dict(color='black', size=45),
                        x=-0.13,
                        y=1.20,
                        showarrow=False,
                        text=f"(d)",
                        textangle=0,
                        yanchor='top',
                        xanchor='left',
                        xref="paper",
                        yref="paper"))
save_figure(fig, f"{path_save}/SupplementaryFigure3/d_1")

table7_sets = [set(x) for x in table7_lists.values()]
table7_tags = [x for x in table7_lists.keys()]
table7_sections = get_sections(table7_sets)
upset_df = pd.DataFrame(index=list(age_acc_df_ctrl.index.values))
for k, v in table7_lists.items():
    upset_df[k] = upset_df.index.isin(v)
upset_df = upset_df.set_index(list(table7_lists.keys()))
fig = upset.UpSet(upset_df, subset_size='count', show_counts=True, min_degree=1, sort_categories_by=None).plot()
plt.savefig(f"{path_save}/SupplementaryFigure3/d_2.png", bbox_inches='tight')
plt.savefig(f"{path_save}/SupplementaryFigure3/d_2.pdf", bbox_inches='tight')
plt.clf()

