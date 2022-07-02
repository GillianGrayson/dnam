import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.python.routines.betas import betas_drop_na
from plotly.subplots import make_subplots
from numpy.ma import masked_array
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import random
import plotly.express as px
import copy
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scripts.python.pheno.datasets.filter import filter_pheno
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict, get_sex_dict
from scripts.python.routines.plot.scatter import add_scatter_trace
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pathlib
from scripts.python.routines.manifest import get_manifest
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout, get_axis
from scripts.python.routines.plot.p_value import add_p_value_annotation
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_absolute_error
import plotly.io as pio
pio.kaleido.scope.mathjax = None
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=False)
from pathlib import Path
from functools import reduce
from scipy.stats import chi2_contingency
from scipy.stats import kruskal, mannwhitneyu
from impyute.imputation.cs import fast_knn, mean, median, random, mice, mode, em


def get_32_electrodes_groups(scale_type):
    if scale_type == "min2max":
        groups = {
            'Alpha_psd': ('Alpha PSD', px.colors.sequential.Reds[0:-1], px.colors.sequential.Reds[0]),
            'Alpha_trp': ('Alpha TRP', px.colors.sequential.Reds[0:-1], px.colors.sequential.Reds[0]),
            'paf': ('Peak Alpha Frequency', px.colors.sequential.Reds[0:-1], px.colors.sequential.Reds[0]),
            'iaf': ('Individual Alpha Frequency', px.colors.sequential.Reds[0:-1], px.colors.sequential.Reds[0]),
            'Beta_psd': ('Beta PSD', px.colors.sequential.Blues[0:-1], px.colors.sequential.Blues[0]),
            'Beta_trp': ('Beta TRP', px.colors.sequential.Blues[0:-1], px.colors.sequential.Blues[0]),
            'Gamma_psd': ('Gamma PSD', px.colors.sequential.Greens[0:-1], px.colors.sequential.Greens[0]),
            'Gamma_trp': ('Gamma TRP', px.colors.sequential.Greens[0:-1], px.colors.sequential.Greens[0]),
            'Theta_psd': ('Theta PSD', px.colors.sequential.Purples[0:-1], px.colors.sequential.Purples[0]),
            'Theta_trp': ('Theta TRP', px.colors.sequential.Purples[0:-1], px.colors.sequential.Purples[0]),
        }
    else:
        groups = {
            'Alpha_psd': ('Alpha PSD', px.colors.diverging.RdBu, "white"),
            'Alpha_trp': ('Alpha TRP', px.colors.diverging.RdBu, "white"),
            'paf': ('Peak Alpha Frequency', px.colors.diverging.RdBu, "white"),
            'iaf': ('Individual Alpha Frequency', px.colors.diverging.RdBu, "white"),
            'Beta_psd': ('Beta PSD', px.colors.diverging.PiYG, "white"),
            'Beta_trp': ('Beta TRP', px.colors.diverging.PiYG, "white"),
            'Gamma_psd': ('Gamma PSD', px.colors.diverging.PuOr, "white"),
            'Gamma_trp': ('Gamma TRP', px.colors.diverging.PuOr, "white"),
            'Theta_psd': ('Theta PSD', px.colors.diverging.BrBG, "white"),
            'Theta_trp': ('Theta TRP', px.colors.diverging.BrBG, "white"),
        }
    return groups


def get_32_electrodes_coordinates():
    coordinates = {
        'Cz': (1.0, 1.0),
        'C3': (0.65, 1.0),
        'C4': (1.35, 1.0),
        'T7': (0.31, 1.0),
        'T8': (1.69, 1.0),
        'Fz': (1.0, 1.365),
        'Pz': (1.0, 0.635),
        'FC1': (0.835, 1.175),
        'FC2': (1.165, 1.175),
        'CP1': (0.835, 0.825),
        'CP2': (1.165, 0.825),
        'F3': (0.712, 1.40),
        'F4': (1.288, 1.40),
        'P3': (0.712, 0.60),
        'P4': (1.288, 0.60),
        'FC5': (0.505, 1.195),
        'FC6': (1.495, 1.195),
        'CP5': (0.505, 0.805),
        'CP6': (1.495, 0.805),
        'F7': (0.445, 1.445),
        'F8': (1.555, 1.445),
        'P7': (0.445, 0.555),
        'P8': (1.555, 0.555),
        'FP1': (0.80, 1.69),
        'FP2': (1.20, 1.69),
        'O1': (0.80, 0.31),
        'O2': (1.20, 0.31),
        'Oz': (1.0, 0.28),
        'FT9': (0.165, 1.245),
        'FT10': (1.835, 1.245),
        'TP9': (0.165, 0.755),
        'TP10': (1.835, 0.755),
    }
    return coordinates



def plot_32_electrodes_scatter(df, column, label, scale_type, path):

    groups = get_32_electrodes_groups(scale_type)
    coordinates = get_32_electrodes_coordinates()

    n_rows = 5
    n_cols = 2
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_yaxes=False, shared_xaxes=False, vertical_spacing=0.0)

    colorbar_xs = [0.44, 0.99]
    colorbar_ys = [0.9, 0.7, 0.5, 0.3, 0.1]

    for g_id, g in enumerate(groups):

        r_id, c_id = divmod(g_id, n_cols)

        df_group = df.loc[df.index.str.contains(g), :]
        df_group["electrode"] = df_group.index.str.replace(f"_{g}", "").values

        xs = [c[0] for c in coordinates.values()]
        ys = [c[1] for c in coordinates.values()]
        elecs = [c for c in coordinates]
        inds_colors = [f"{c}_{g}" for c in coordinates]
        colors = df_group.loc[inds_colors, column].values

        is_the_same_colors = False
        if len(set(colors)) == 1:
            is_the_same_colors = True

        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=0.3, y0=0.28, x1=1.7, y1=1.72,
             line={
                'color': "black",
                'dash': 'dot',
                'width': 0.5
            },
            layer="below",
            row=r_id + 1,
            col=c_id + 1
        )
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=0.13, y0=0.09, x1=1.87, y1=1.91,
            line={
                'color': "black",
                'dash': 'solid',
                'width': 0.5
            },
            layer="below",
            row=r_id + 1,
            col=c_id + 1
        )
        fig.add_shape(
            type="line",
            xref="x", yref="y",
            x0=0.13, y0=1, x1=1.87, y1=1,
            line={
                'color': "black",
                'dash': 'dot',
                'width': 0.5
            },
            layer="below",
            row=r_id + 1,
            col=c_id + 1
        )
        fig.add_shape(
            type="line",
            xref="x", yref="y",
            x0=1, y0=0.09, x1=1, y1=1.91,
            line={
                'color': "black",
                'dash': 'dot',
                'width': 0.5
            },
            layer="below",
            row=r_id + 1,
            col=c_id + 1
        )
        fig.add_annotation(
            x=1, y=1.99,
            text="NASION",
            showarrow=False,
            font=dict(color='black', size=20),
            row=r_id + 1,
            col=c_id + 1
        )
        fig.add_annotation(
            x=1, y=2.2,
            text=f"{groups[g][0]}",
            showarrow=False,
            font=dict(color='black', size=40),
            row=r_id + 1,
            col=c_id + 1
        )
        fig.add_annotation(
            x=1, y=1.815,
            text="FRONTAL",
            showarrow=False,
            font=dict(color='black', size=15),
            row=r_id + 1,
            col=c_id + 1
        )
        fig.add_annotation(
            x=1, y=0,
            text="INION",
            showarrow=False,
            font=dict(color='black', size=20),
            row=r_id + 1,
            col=c_id + 1
        )
        fig.add_annotation(
            x=0.07, y=1.43,
            text="LEFT",
            showarrow=False,
            font=dict(color='black', size=20),
            row=r_id + 1,
            col=c_id + 1
        )
        fig.add_annotation(
            x=1.93, y=1.43,
            text="RIGHT",
            showarrow=False,
            font=dict(color='black', size=20),
            row=r_id + 1,
            col=c_id + 1
        )

        if not is_the_same_colors:
            if scale_type == "min2max":
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        showlegend=False,
                        mode='markers+text',
                        marker=dict(
                            size=37,
                            opacity=1,
                            line=dict(
                                width=1
                            ),
                            color=colors,
                            colorscale=groups[g][1],
                            showscale=True,
                            colorbar=dict(
                                title=dict(text="", font=dict(size=20)), tickfont=dict(size=20),
                                x=colorbar_xs[c_id],
                                y=colorbar_ys[r_id],
                                len=0.13
                            )
                        ),
                        text=elecs,
                        textposition="bottom center",
                        textfont=dict(
                            family="arial",
                            size=18,
                            color="Black"
                        ),
                    ),
                    row=r_id + 1,
                    col=c_id + 1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        showlegend=False,
                        mode='markers+text',
                        marker=dict(
                            cmid=0,
                            size=37,
                            opacity=1,
                            line=dict(
                                width=1
                            ),
                            color=colors,
                            colorscale=groups[g][1],
                            showscale=True,
                            colorbar=dict(
                                title=dict(text="", font=dict(size=20)), tickfont=dict(size=20),
                                x=colorbar_xs[c_id],
                                y=colorbar_ys[r_id],
                                len=0.13
                            )
                        ),
                        text=elecs,
                        textposition="bottom center",
                        textfont=dict(
                            family="arial",
                            size=18,
                            color="Black"
                        ),
                    ),
                    row=r_id + 1,
                    col=c_id + 1
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    showlegend=False,
                    mode='markers+text',
                    marker=dict(
                        size=37,
                        opacity=1,
                        line=dict(
                            width=1
                        ),
                        color=groups[g][2],
                        colorscale=groups[g][1],
                        showscale=False,
                        colorbar=dict(
                            title=dict(text="", font=dict(size=20)), tickfont=dict(size=20),
                            x=colorbar_xs[c_id],
                            y=colorbar_ys[r_id],
                            len=0.13
                        )
                    ),
                    text=elecs,
                    textposition="bottom center",
                    textfont=dict(
                        family="arial",
                        size=18,
                        color="Black"
                    ),
                ),
                row=r_id + 1,
                col=c_id + 1
            )

        fig.update_xaxes(
            row=r_id + 1,
            col=c_id + 1,
            autorange=False,
            range=[-0.2, 2.2],
            visible=False,
            title_text=f"",
            showgrid=False,
            zeroline=False,
            linecolor='black',
            showline=False,
            gridcolor='gainsboro',
            gridwidth=0.05,
            mirror=False,
            ticks='outside',
            titlefont=dict(
                color='black',
                size=20
            ),
            showticklabels=False,
            tickangle=0,
            tickfont=dict(
                color='black',
                size=20
            ),
            exponentformat='e',
            showexponent='all'
        )
        fig.update_yaxes(
            row=r_id + 1,
            col=c_id + 1,
            autorange=False,
            range=[-0.2, 2.2],
            visible=False,
            title_text="",
            showgrid=False,
            zeroline=False,
            linecolor='black',
            showline=False,
            gridcolor='gainsboro',
            gridwidth=0.05,
            mirror=False,
            ticks='outside',
            titlefont=dict(
                color='black',
                size=20
            ),
            showticklabels=False,
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
            font=dict(size=60)
        ),
        template="none",
        autosize=False,
        width=1800,
        height=3500,
        margin=go.layout.Margin(
            l=100,
            r=100,
            b=0,
            t=60,
            pad=0
        )
    )
    fig.update_layout(legend_font_size=50)
    fig.update_layout(legend={'itemsizing': 'constant'})
    save_figure(fig, f"{path}/{column}_scatter")


def plot_32_electrodes_beeswarm(df_x, df_color, samples, label, path):

    n_top = 30

    df_x = df_x.head(n_top)
    df_x = df_x.loc[::-1]

    fig = go.Figure()

    for feat_id, feat in enumerate(df_x.index.values):
        showscale = True if feat_id == 0 else False
        xs = df_x.loc[feat, samples].values
        colors = df_color.loc[samples, feat].values

        N = len(xs)
        row_height = 0.45
        nbins = 20
        xs = list(xs)
        xs = np.array(xs, dtype=float)
        quant = np.round(nbins * (xs - np.min(xs)) / (np.max(xs) - np.min(xs) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        ys = feat_id + ys

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                showlegend=False,
                mode='markers',
                marker=dict(
                    size=20,
                    opacity=0.7,
                    line=dict(
                        width=0.00
                    ),
                    color=colors,
                    colorscale=px.colors.sequential.Rainbow,
                    showscale=showscale,
                    colorbar=dict(
                        title=dict(text="", font=dict(size=50)),
                        tickfont=dict(size=50),
                        tickmode="array",
                        tickvals=[min(colors), max(colors)],
                        ticktext=["Min", "Max"],
                        x=1.03,
                        y=0.5,
                        len=0.99
                    )
                ),
            )
        )

    add_layout(fig, label, "", f"")
    fig.update_layout(legend_font_size=20)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey')
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(df_x.index.values))),
            ticktext=df_x.index.values
        )
    )
    fig.update_yaxes(autorange=False)
    fig.update_layout(yaxis_range=[-0.5, len(df_x.index.values) - 0.5])
    fig.update_yaxes(tickfont_size=35)
    fig.update_xaxes(tickfont_size=50)
    fig.update_xaxes(title_font_size=50)
    fig.update_xaxes(nticks=6)
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    fig.update_layout(
        autosize=False,
        width=1900,
        height=3400,
        margin=go.layout.Margin(
            l=350,
            r=100,
            b=130,
            t=20,
            pad=0
        )
    )
    save_figure(fig, f"{path}/beeswarm")


def plot_32_electrodes_bar(df, column, label, scale_type, path):

    n_top = 30

    groups = get_32_electrodes_groups(scale_type)

    for g_id, g in enumerate(groups):
        df_group = df.loc[df.index.str.contains(g), :]
        df.loc[df_group.index, 'color'] = groups[g][1][-2]

    df_bar = df.head(n_top)
    xs = df_bar.loc[:, column].values[::-1]
    ys = df_bar.index.values[::-1]
    colors = df_bar.loc[:, 'color'].values[::-1]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=xs,
            y=list(range(len(ys))),
            orientation='h',
            marker=dict(color=colors, opacity=1.0)
        )
    )
    add_layout(fig, label, "", f"")
    fig.update_layout(legend_font_size=20)
    fig.update_layout(showlegend=False)
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(ys))),
            ticktext=ys
        )
    )
    fig.update_yaxes(autorange=False)
    fig.update_layout(yaxis_range=[-0.5, len(ys) - 0.5])
    fig.update_yaxes(tickfont_size=35)
    fig.update_xaxes(tickfont_size=50)
    fig.update_xaxes(title_font_size=50)
    fig.update_xaxes(nticks=6)
    fig.update_layout(
        autosize=False,
        width=1200,
        height=3400,
        margin=go.layout.Margin(
            l=350,
            r=20,
            b=130,
            t=20,
            pad=0
        )
    )
    save_figure(fig, f"{path}/{column}_bar")