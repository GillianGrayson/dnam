import pandas as pd
import numpy as np
from scipy.stats import kruskal
from scripts.python.routines.manifest import get_manifest
import pathlib
from statsmodels.stats.multitest import multipletests
from scripts.python.routines.plot.layout import add_layout, get_axis
from scripts.python.routines.plot.save import save_figure
import plotly.graph_objects as go
from tqdm import tqdm
from scripts.python.routines.mvals import logit2, expit2


def KW_Control(datasets, manifest, df, cpgs, path, y_label):
    cpgs_metrics_dict = {'CpG': cpgs}

    cpgs_metrics_dict['KW_Controls_pval'] = []
    for cpg_id, cpg in enumerate(tqdm(cpgs)):
        kw_vals = {}
        for dataset in datasets:
            vals_i = df.loc[(df['Status'] == 'Control') & (df['Dataset'] == dataset), cpg].values
            kw_vals[dataset] = vals_i
        stat, pval = kruskal(*kw_vals.values())
        cpgs_metrics_dict['KW_Controls_pval'].append(pval)
    _, pvals_corr, _, _ = multipletests(cpgs_metrics_dict['KW_Controls_pval'], 0.05, method='fdr_bh')
    cpgs_metrics_dict['KW_Controls_pval_fdr_bh'] = pvals_corr
    _, pvals_corr, _, _ = multipletests(cpgs_metrics_dict['KW_Controls_pval'], 0.05, method='bonferroni')
    cpgs_metrics_dict['KW_Controls_pval_bonferroni'] = pvals_corr
    cpgs_metrics_df = pd.DataFrame(cpgs_metrics_dict)
    cpgs_metrics_df.set_index('CpG', inplace=True)
    cpgs_metrics_df.sort_values(['KW_Controls_pval_fdr_bh'], ascending=[False], inplace=True)
    cpgs_metrics_df.to_excel(f"{path}/cpgs_metrics.xlsx", index=True)

    cpgs_to_plot_df = cpgs_metrics_df.head(20)
    for cpg_id, (cpg, row) in enumerate(cpgs_to_plot_df.iterrows()):
        pval = row['KW_Controls_pval_fdr_bh']
        gene = manifest.at[cpg, 'Gene']

        dist_num_bins = 30
        fig = go.Figure()
        for dataset in datasets:
            vals_i = df.loc[(df['Status'] == 'Control') & (df['Dataset'] == dataset), cpg].values
            fig.add_trace(
                go.Violin(
                    y=vals_i,
                    name=dataset,
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=False,
                    marker=dict(line=dict(width=0.3), opacity=0.8),
                    points='all',
                    bandwidth=np.ptp(vals_i) / dist_num_bins,
                    opacity=0.8
                )
            )
        add_layout(fig, "", y_label, f"{gene}<br>p-value: {pval:0.2e}")
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
                y=1.25,
                xanchor="center",
                x=0.5
            )
        )
        save_figure(fig, f"{path}/fig/{cpg_id:3d}_{cpg}")

    return cpgs_metrics_df
