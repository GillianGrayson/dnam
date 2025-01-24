import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scripts.python.routines.plot.layout import add_layout
from scripts.python.routines.plot.save import save_figure
import plotly.graph_objects as go
from tqdm import tqdm
import plotly.express as px


def perform_test_for_controls(datasets, manifest, df, cpgs, path, y_label, n_plot=20):
    cpgs_metrics_dict = {'features': cpgs}
    cpgs_metrics_dict['pval'] = []
    for cpg_id, cpg in enumerate(tqdm(cpgs)):
        vals = {}
        for dataset in datasets:
            vals_i = df.loc[(df['Status'] == 'Control') & (df['Dataset'] == dataset), cpg].values
            vals[dataset] = vals_i
        if len(datasets) > 2:
            stat, pval = kruskal(*vals.values())
        elif len(datasets) == 2:
            stat, pval = mannwhitneyu(*vals.values(), alternative='two-sided')
        else:
            raise ValueError("Number of datasets less than 2")
        cpgs_metrics_dict['pval'].append(pval)
    _, pvals_corr, _, _ = multipletests(cpgs_metrics_dict['pval'], 0.05, method='fdr_bh')
    cpgs_metrics_dict['pval_fdr_bh'] = pvals_corr
    _, pvals_corr, _, _ = multipletests(cpgs_metrics_dict['pval'], 0.05, method='bonferroni')
    cpgs_metrics_dict['pval_bonferroni'] = pvals_corr
    cpgs_metrics_df = pd.DataFrame(cpgs_metrics_dict)
    cpgs_metrics_df.set_index('features', inplace=True)
    cpgs_metrics_df.sort_values(['pval_fdr_bh'], ascending=[False], inplace=True)

    cpgs_to_plot_df = cpgs_metrics_df.head(n_plot)
    for cpg_id, (cpg, row) in enumerate(cpgs_to_plot_df.iterrows()):
        pval = row['pval_fdr_bh']
        gene = manifest.at[cpg, 'Gene']

        dist_num_bins = 25
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
        add_layout(fig, "", y_label, f"{cpg} ({gene})<br>p-value: {pval:0.2e}")
        fig.update_layout({'colorway': px.colors.qualitative.Set1})
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
        save_figure(fig, f"{path}/{cpg_id:03d}_{cpg}")

    return cpgs_metrics_df
