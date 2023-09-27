import numpy as np
import plotly.io as pio
pio.kaleido.scope.mathjax = None
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import missingno as msno


def add_iqr_outs_to_df(df, df_train, feats):
    out_columns = []
    for f in feats:
        q1 = df_train[f].quantile(0.25)
        q3 = df_train[f].quantile(0.75)
        iqr = q3 - q1
        df[f"{f}_out_iqr"] = True
        out_columns.append(f"{f}_out_iqr")
        filter = (df[f] >= q1 - 1.5 * iqr) & (df[f] <= q3 + 1.5 * iqr)
        df.loc[filter, f"{f}_out_iqr"] = False
    df[f"n_outs_iqr"] = df.loc[:, out_columns].sum(axis=1)


def plot_iqr_plots(df, feats, color, title, path):

    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

    # Plot hist for Number of IQR outliers
    hist_bins = np.linspace(-0.5, len(feats) + 0.5, len(feats) + 2)
    fig = plt.figure(figsize=(4, 3))
    sns.set_theme(style='whitegrid')
    histplot = sns.histplot(
        data=df,
        x=f"n_outs_iqr",
        multiple="stack",
        bins=hist_bins,
        edgecolor='k',
        linewidth=1,
        color=color,
    )
    histplot.set(xlim=(-0.5, len(feats) + 0.5))
    histplot.set_title(title)
    histplot.set_xlabel("Number of IQR outliers")
    plt.savefig(f"{path}/hist_nOuts.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/hist_nOuts.pdf", bbox_inches='tight')
    plt.close(fig)

    # Prepare dataframe for msno lib
    out_columns = [f"{f}_out_iqr" for f in feats]
    df_msno = df.loc[:, out_columns].copy()
    df_msno.replace({True: np.nan}, inplace=True)
    df_msno.rename(columns=dict(zip(out_columns, feats)), inplace=True)

    # Plot barplot for features with outliers
    msno_bar = msno.bar(
        df=df_msno,
        label_rotation=90,
        color=color,
        figsize=(0.4 * len(feats), 4)
    )
    plt.xticks(ha='center')
    plt.setp(msno_bar.xaxis.get_majorticklabels(), ha="center")
    msno_bar.set_title(title, fontdict={'fontsize': 22})
    msno_bar.set_ylabel("Non-outlier samples", fontdict={'fontsize': 22})
    plt.savefig(f"{path}/bar_feats_nOuts.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/bar_feats_nOuts.pdf", bbox_inches='tight')
    plt.clf()

    # Plot matrix of samples outliers distribution
    msno_mtx = msno.matrix(
        df=df_msno,
        label_rotation=90,
        color=colors.to_rgb(color),
        figsize=(0.7 * len(feats), 5)
    )
    plt.xticks(ha='center')
    plt.setp(msno_bar.xaxis.get_majorticklabels(), ha="center")
    msno_mtx.set_title(title, fontdict={'fontsize': 22})
    msno_mtx.set_ylabel("Samples", fontdict={'fontsize': 22})
    plt.savefig(f"{path}/matrix_featsOuts.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/matrix_featsOuts.pdf", bbox_inches='tight')
    plt.clf()

    # Plot heatmap of features outliers correlations
    msno_heatmap = msno.heatmap(
        df=df_msno,
        label_rotation=90,
        cmap="bwr",
        fontsize=12,
        figsize=(0.6 * len(feats), 0.6 * len(feats))
    )
    msno_heatmap.set_title(title, fontdict={'fontsize': 22})
    plt.setp(msno_heatmap.xaxis.get_majorticklabels(), ha="center")
    msno_heatmap.collections[0].colorbar.ax.tick_params(labelsize=20)
    plt.savefig(f"{path}/heatmap_featsOutsCorr.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/heatmap_featsOutsCorr.pdf", bbox_inches='tight')
    plt.clf()
