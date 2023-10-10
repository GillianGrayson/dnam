import numpy as np
import plotly.io as pio
pio.kaleido.scope.mathjax = None
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import missingno as msno
from tqdm import tqdm
import plotly.express as px
import patchworklib as pw
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from sklearn.metrics import mean_absolute_error
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.lines as mlines


def plot_cls_dim_red(df, col_class, cls_names, col_prob, cols_dim_red, title, fn):
    norm = plt.Normalize(df[col_prob].min(), df[col_prob].max())
    sm = plt.cm.ScalarMappable(cmap="seismic", norm=norm)
    sm.set_array([])
    df.loc[df[col_class] == cls_names[0], 'Symbol'] = 'o'
    df.loc[df[col_class] == cls_names[1], 'Symbol'] = 'X'

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.set_theme(style='whitegrid')
    scatter = sns.scatterplot(
        data=df,
        x=cols_dim_red[0],
        y=cols_dim_red[1],
        palette='seismic',
        hue=col_prob,
        linewidth=0,
        alpha=0.6,
        edgecolor="k",
        style=df.loc[:, 'Symbol'].values,
        s=25,
        ax=ax
    )
    scatter.get_legend().remove()
    scatter.figure.colorbar(sm, label=col_prob)
    scatter.set_title(title, loc='left', y=1.05, fontdict={'fontsize': 20})

    legend_handles = [
        mlines.Line2D([], [], marker='o', linestyle='None', markeredgecolor='k', markerfacecolor='lightgrey', markersize=10, label=cls_names[0]),
        mlines.Line2D([], [], marker='X', linestyle='None', markeredgecolor='k', markerfacecolor='lightgrey', markersize=10, label=cls_names[1])
    ]
    plt.legend(handles=legend_handles, title="Samples", bbox_to_anchor=(0.4, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=2, frameon=False)

    plt.savefig(f"{fn}.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{fn}.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_regression_error_distributions(df, feats, color, title, path, col_error_abs):

    q25, q50, q75 = np.percentile(df[col_error_abs].values, [25, 50, 75])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.set_theme(style='whitegrid')
    kdeplot = sns.kdeplot(
        data=df,
        x=col_error_abs,
        color=color,
        linewidth=4,
        cut=0,
        ax=ax
    )
    kdeplot.set_title(title)
    kdeline = ax.lines[0]
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    ax.vlines(q50, 0, np.interp(q50, xs, ys), color='black', ls=':')
    ax.fill_between(xs, 0, ys, where=(q25 <= xs) & (xs <= q75), facecolor=color, alpha=0.9)
    ax.fill_between(xs, 0, ys, where=(xs <= q25), interpolate=True, facecolor='dodgerblue', alpha=0.9)
    ax.fill_between(xs, 0, ys, where=(xs >= q75), interpolate=True, facecolor='crimson', alpha=0.9)
    plt.savefig(f"{path}/kde_error_abs.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/kde_error_abs.pdf", bbox_inches='tight')
    plt.close(fig)

    df_fig = df.loc[(df[col_error_abs] <= q25) | (df[col_error_abs] >= q75), list(feats) + [col_error_abs]].copy()
    df_fig.loc[df[col_error_abs] <= q25, 'abs(Error)'] = '<q25'
    df_fig.loc[df[col_error_abs] >= q75, 'abs(Error)'] = '>q75'

    df_stat = pd.DataFrame(index=list(feats))
    for feat in list(feats):
        vals = {}
        for group in ['<q25', '>q75']:
            vals[group] = df_fig.loc[df_fig['abs(Error)'] == group, feat].values
            df_stat.at[feat, f"mean_{group}"] = np.mean(vals[group])
            df_stat.at[feat, f"median_{group}"] = np.median(vals[group])
            df_stat.at[feat, f"q75_{group}"], df_stat.at[feat, f"q25_{group}"] = np.percentile(vals[group], [75, 25])
            df_stat.at[feat, f"iqr_{group}"] = df_stat.at[feat, f"q75_{group}"] - df_stat.at[feat, f"q25_{group}"]
        _, df_stat.at[feat, "mw_pval"] = mannwhitneyu(vals['<q25'], vals['>q75'], alternative='two-sided')

    _, df_stat.loc[feats, "mw_pval_fdr_bh"], _, _ = multipletests(df_stat.loc[feats, "mw_pval"], 0.05, method='fdr_bh')
    df_stat.sort_values([f"mw_pval_fdr_bh"], ascending=[True], inplace=True)
    df_stat.to_excel(f"{path}/feats_stat.xlsx", index_label='Features')

    feats_sorted = df_stat.index.values
    axs = {}
    pw_rows = []
    n_cols = 5
    n_rows = int(np.ceil(len(feats_sorted) / n_cols))
    for r_id in range(n_rows):
        pw_cols = []
        for c_id in range(n_cols):
            rc_id = r_id * n_cols + c_id
            if rc_id < len(feats_sorted):
                feat = feats_sorted[rc_id]
                axs[feat] = pw.Brick(figsize=(3, 4))
                sns.set_theme(style='whitegrid')
                sns.violinplot(
                    data=df_fig,
                    x='abs(Error)',
                    y=feat,
                    palette={'<q25': 'dodgerblue', '>q75': 'crimson'},
                    scale='width',
                    order=['<q25', '>q75'],
                    saturation=0.75,
                    cut=0,
                    ax=axs[feat]
                )
                mw_pval = df_stat.at[feat, "mw_pval_fdr_bh"]
                pval_formatted = [f'{mw_pval:.2e}']
                annotator = Annotator(
                    axs[feat],
                    pairs=[('<q25', '>q75')],
                    data=df_fig,
                    x='abs(Error)',
                    y=feat,
                    order=['<q25', '>q75'],
                )
                annotator.set_custom_annotations(pval_formatted)
                annotator.configure(loc='outside')
                annotator.annotate()
                pw_cols.append(axs[feat])
            else:
                empty_fig = pw.Brick(figsize=(3.6, 4))
                empty_fig.axis('off')
                pw_cols.append(empty_fig)

        pw_rows.append(pw.stack(pw_cols, operator="|"))
    pw_fig = pw.stack(pw_rows, operator="/")
    pw_fig.savefig(f"{path}/feats_violins.png", bbox_inches='tight', dpi=200)
    pw_fig.savefig(f"{path}/feats_violins.pdf", bbox_inches='tight')



