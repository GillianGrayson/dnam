import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import sys
from matplotlib.colors import ListedColormap
from random import Random
import plotly.express as px


def geneplot_mhat(df, markeridcol, chr, pv, gwasp, markernames, gfont, gstyle, ax):
    if markeridcol is not None:
        if markernames is not None and markernames is True:
            for i in df[markeridcol].unique():
                if df.loc[df[markeridcol] == i, pv].iloc[0] <= gwasp:
                    if gstyle == 1:
                        plt.text(df.loc[df[markeridcol] == i, 'ind'].iloc[0], df.loc[df[markeridcol] == i, 'tpval'].iloc[0],
                                str(i), fontsize=gfont)
                    elif gstyle == 2:
                        plt.annotate(i, xy=(df.loc[df[markeridcol] == i, 'ind'].iloc[0], df.loc[df[markeridcol] == i, 'tpval'].iloc[0]),
                                     xycoords='data', xytext=(5, -15), textcoords='offset points', size=6,
                                     bbox=dict(boxstyle="round", alpha=0.2),
                                     arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.2, relpos=(0, 0)))
        elif markernames is not None and isinstance(markernames, (tuple, list)):
            for i in df[markeridcol].unique():
                if i in markernames:
                    if gstyle == 1:
                        plt.text(df.loc[df[markeridcol] == i, 'ind'].iloc[0], df.loc[df[markeridcol] == i, 'tpval'].iloc[0],
                            str(i), fontsize=gfont)
                    elif gstyle == 2:
                        plt.annotate(i, xy=(df.loc[df[markeridcol] == i, 'ind'].iloc[0], df.loc[df[markeridcol] == i, 'tpval'].iloc[0]),
                                     xycoords='data', xytext=(5, -15), textcoords='offset points', size=6,
                                     bbox=dict(boxstyle="round", alpha=0.2),
                                     arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.2, relpos=(0, 0)))
        elif markernames is not None and isinstance(markernames, dict):
            for i in df[markeridcol].unique():
                if i in markernames:
                    if gstyle == 1:
                        plt.text(df.loc[df[markeridcol] == i, 'ind'].iloc[0], df.loc[df[markeridcol] == i, 'tpval'].iloc[0],
                             markernames[i], fontsize=gfont)
                    elif gstyle == 2:
                        plt.annotate(markernames[i], xy=(
                        df.loc[df[markeridcol] == i, 'ind'].iloc[0], df.loc[df[markeridcol] == i, 'tpval'].iloc[0]),
                                     xycoords='data', xytext=(5, -15), textcoords='offset points', size=6,
                                     bbox=dict(boxstyle="round", alpha=0.2),
                                     arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.2, relpos=(0, 0)))
    else:
        raise Exception("provide 'markeridcol' parameter")


def check_for_nonnumeric(pd_series=None):
    if pd.to_numeric(pd_series, errors='coerce').isna().sum() == 0:
        return 0
    else:
        return 1


def gene_plot(d, geneid, lfc, lfc_thr, pv_thr, genenames, gfont, pv, gstyle):
    if genenames is not None and genenames == "deg":
        for i in d[geneid].unique():
            if (d.loc[d[geneid] == i, lfc].iloc[0] >= lfc_thr[0] and d.loc[d[geneid] == i, pv].iloc[0] < pv_thr[0]) or \
                    (d.loc[d[geneid] == i, lfc].iloc[0] <= -lfc_thr[1] and d.loc[d[geneid] == i, pv].iloc[0] < pv_thr[1]):
                if gstyle == 1:
                    plt.text(d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0], i,
                                  fontsize=gfont)
                elif gstyle == 2:
                    plt.annotate(i, xy=(d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0]),
                                 xycoords='data', xytext=(5, -15), textcoords='offset points', size=6,
                                 bbox=dict(boxstyle="round", alpha=0.1),
                                 arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1, relpos=(0, 0)))
                else:
                    print("Error: invalid gstyle choice")
                    sys.exit(1)
    elif genenames is not None and type(genenames) is tuple:
        for i in d[geneid].unique():
            if i in genenames:
                if gstyle == 1:
                    plt.text(d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0], i,
                                  fontsize=gfont)
                elif gstyle == 2:
                    plt.annotate(i, xy=(d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0]),
                                 xycoords='data', xytext=(5, -15), textcoords='offset points', size=6,
                                 bbox=dict(boxstyle="round", alpha=0.1),
                                 arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1, relpos=(0, 0)))
                else:
                    print("Error: invalid gstyle choice")
                    sys.exit(1)
    elif genenames is not None and type(genenames) is dict:
        for i in d[geneid].unique():
            if i in genenames:
                if gstyle == 1:
                    plt.text(d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0],
                                  genenames[i], fontsize=gfont)
                elif gstyle == 2:
                    plt.annotate(genenames[i], xy=(d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0]),
                                 xycoords='data', xytext=(5, -15), textcoords='offset points', size=6,
                                 bbox=dict(boxstyle="round", alpha=0.1),
                                 arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1, relpos=(0, 0)))
                else:
                    print("Error: invalid gstyle choice")
                    sys.exit(1)

def mhat(df="dataframe", chr=None, pv=None, log_scale=True, dim=(6,4), ar=90, gwas_sign_line=False,
         gwasp=5E-08, dotsize=1, markeridcol=None, markernames=None, gfont=8, valpha=1,
         axxlabel=None, axylabel=None, axlabelfontsize=9, axlabelfontname="Arial", axtickfontsize=6,
         axtickfontname="Arial", gstyle=1, figname='manhattan', theme=None, path=''):

    _x, _y = 'Chromosomes', r'$ -\log_{10}(\mathrm{p-value})$'

    if log_scale:
        # minus log10 of P-value
        df['tpval'] = -np.log10(df[pv].values)
    else:
        # for Fst values
        df['tpval'] = df[pv]
    # df = df.sort_values(chr)
    # if the column contains numeric strings
    df = df.loc[pd.to_numeric(df[chr], errors='coerce').sort_values().index]
    # add indices
    df['ind'] = range(len(df))

    color_list = px.colors.qualitative.Dark24
    xlabels = []
    xticks = []
    if theme == 'dark':
        plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=dim)
    i = 0
    for label, df1 in df.groupby(chr):
        df1.plot(kind='scatter', x='ind', y='tpval', color=color_list[i], s=dotsize, alpha=valpha, ax=ax)
        df1_max_ind = df1['ind'].iloc[-1]
        df1_min_ind = df1['ind'].iloc[0]
        xlabels.append(label)
        xticks.append((df1_max_ind - (df1_max_ind - df1_min_ind) / 2))
        i += 1

    # add GWAS significant line
    if gwas_sign_line is True:
        ax.axhline(y=-np.log10(gwasp), linestyle='--', color='black', linewidth=1)
    if markernames is not None:
        geneplot_mhat(df, markeridcol, chr, pv, gwasp, markernames, gfont, gstyle, ax=ax)
    ax.margins(x=0)
    ax.margins(y=0)
    ax.set_xticks(xticks)
    if log_scale:
        ax.set_ylim([0, max(df['tpval'] + 1)])
    plt.grid(visible=False, axis='x')
    ax.set_xticklabels(xlabels, rotation=ar, fontsize=axtickfontsize, fontname=axtickfontname)
    if axxlabel:
        _x = axxlabel
    if axylabel:
        _y = axylabel
    ax.set_xlabel(_x, fontsize=axlabelfontsize, fontname=axlabelfontname)
    ax.set_ylabel(_y, fontsize=axlabelfontsize, fontname=axlabelfontname)
    plt.savefig(f"{path}/{figname}.png", bbox_inches='tight', dpi=400)
    plt.savefig(f"{path}/{figname}.pdf", bbox_inches='tight', dpi=400)
    plt.clf()
    plt.close()

def volcano(df="dataframe", lfc=None, pv=None, lfc_thr=(1, 1), pv_thr=(0.05, 0.05), color=("green", "grey", "red"),
            valpha=1, geneid=None, genenames=None, gfont=8, dim=(5, 5), ar=90, dotsize=1, markerdot="o",
            sign_line=False, gstyle=1, axtickfontsize=9,
            axtickfontname="Arial", axlabelfontsize=9, axlabelfontname="Arial", axxlabel=None,
            axylabel=None, xlm=None, ylm=None, plotlegend=False, legendpos='best',
            figname='volcano', legendanchor=None,
            legendlabels=['Significant up', 'Not significant', 'Significant down'], theme=None, path=''):
    _x = r'$ \log_{2}(\mathrm{Fold Change})$'
    _y = r'$ -\log_{10}(\mathrm{p-value})$'
    color = color
    # check if dataframe contains any non-numeric character
    assert check_for_nonnumeric(df[lfc]) == 0, 'dataframe contains non-numeric values in lfc column'
    assert check_for_nonnumeric(df[pv]) == 0, 'dataframe contains non-numeric values in pv column'
    # this is important to check if color or logpv exists and drop them as if you run multiple times same command
    # it may update old instance of df
    df = df.drop(['color_add_axy', 'logpv_add_axy'], axis=1, errors='ignore')
    assert len(set(color)) == 3, 'unique color must be size of 3'
    df.loc[(df[lfc] >= lfc_thr[0]) & (df[pv] < pv_thr[0]), 'color_add_axy'] = color[0]  # upregulated
    df.loc[(df[lfc] <= -lfc_thr[1]) & (df[pv] < pv_thr[1]), 'color_add_axy'] = color[2]  # downregulated
    df['color_add_axy'].fillna(color[1], inplace=True)  # intermediate
    df['logpv_add_axy'] = -(np.log10(np.array(df[pv].values.astype(float))))
    # plot
    assign_values = {col: i for i, col in enumerate(color)}
    color_result_num = [assign_values[i] for i in df['color_add_axy']]
    #assert len(set(color_result_num)) == 3, \
    #    'either significant or non-significant genes are missing; try to change lfc_thr or pv_thr to include ' \
    #    'both significant and non-significant genes'
    if theme == 'dark':
        plt.style.use('dark_background')
    plt.subplots(figsize=dim)
    if plotlegend:
        s = plt.scatter(df[lfc], df['logpv_add_axy'], c=color_result_num, cmap=ListedColormap(color), alpha=valpha,
                        s=dotsize, marker=markerdot)
        assert len(legendlabels) == 3, 'legendlabels must be size of 3'
        plt.legend(handles=s.legend_elements()[0], labels=legendlabels, loc=legendpos, bbox_to_anchor=legendanchor)
    else:
        plt.scatter(df[lfc], df['logpv_add_axy'], c=color_result_num, cmap=ListedColormap(color), alpha=valpha,
                    s=dotsize, marker=markerdot)
    if sign_line:
        plt.axhline(y=-np.log10(pv_thr[0]), linestyle='--', color='black', linewidth=1)
        plt.axvline(x=lfc_thr[0], linestyle='--', color='black', linewidth=1)
        plt.axvline(x=-lfc_thr[1], linestyle='--', color='black', linewidth=1)
    gene_plot(df, geneid, lfc, lfc_thr, pv_thr, genenames, gfont, pv, gstyle)

    if axxlabel:
        _x = axxlabel
    if axylabel:
        _y = axylabel

    plt.xlabel(_x, fontsize=axlabelfontsize, fontname=axlabelfontname)
    plt.ylabel(_y, fontsize=axlabelfontsize, fontname=axlabelfontname)
    if xlm:
        plt.xlim(left=xlm[0], right=xlm[1])
        plt.xticks(fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)

    else:
        plt.xticks(fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
    if ylm:
        plt.ylim(bottom=ylm[0], top=ylm[1])
        plt.yticks(np.arange(ylm[0], ylm[1], ylm[2]),  fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
    else:
        plt.yticks(fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
    plt.savefig(f"{path}/{figname}.png", bbox_inches='tight', dpi=400)
    plt.savefig(f"{path}/{figname}.pdf", bbox_inches='tight', dpi=400)
    plt.clf()
    plt.close()
