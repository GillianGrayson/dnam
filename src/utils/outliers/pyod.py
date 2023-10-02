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


def add_pyod_outs_to_df(df, pyod_methods, feats):
    for method_name, method in pyod_methods.items():
        print(method_name)
        X = df.loc[:, feats].values
        df[f"{method_name}"] = method.predict(X)
        df[f"{method_name} anomaly score"] = method.decision_function(X)
        probability = method.predict_proba(X)
        df[f"{method_name} probability inlier"] = probability[:, 0]
        df[f"{method_name} probability outlier"] = probability[:, 1]
        df[f"{method_name} confidence"] = method.predict_confidence(X)
    df["Detections"] = df.loc[:, [f"{method}" for method in pyod_methods]].sum(axis=1)


def plot_pyod_outs(df, pyod_methods, color, title, path):
    # Plot hist for Number of detections as outlier in different PyOD methods
    hist_bins = np.linspace(-0.5, len(pyod_methods) + 0.5, len(pyod_methods) + 2)
    fig = plt.figure()
    sns.set_theme(style='whitegrid')
    histplot = sns.histplot(
        data=df,
        x=f"Detections",
        multiple="stack",
        bins=hist_bins,
        discrete=True,
        edgecolor='k',
        linewidth=0.05,
        color=color,
    )
    histplot.set(xlim=(-0.5, len(pyod_methods) + 0.5))
    histplot.set_title(title)
    histplot.set_xlabel("Number of detections as outlier in different methods")
    plt.savefig(f"{path}/hist_nDetections.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/hist_nDetections.pdf", bbox_inches='tight')
    plt.close(fig)

    # Plot metrics distribution for each method
    metrics = {
        'anomaly score': 'AnomalyScore',
        'probability outlier': 'Probability',
        'confidence': 'Confidence'
    }
    colors_methods = {m: px.colors.qualitative.Alphabet[m_id] for m_id, m in enumerate(pyod_methods)}
    for m_name, m_title in metrics.items():
        n_cols = 6
        n_rows = int(np.ceil(len(pyod_methods) / n_cols))
        method_names = list(pyod_methods.keys())
        pw_rows = []
        for r_id in range(n_rows):
            pw_cols = []
            for c_id in range(n_cols):
                rc_id = r_id * n_cols + c_id
                if rc_id < len(pyod_methods):
                    method = method_names[rc_id]
                    brick = pw.Brick(figsize=(1.5, 2))
                    sns.set_theme(style='whitegrid')
                    data_fig = df[f"{method} {m_name}"].values
                    sns.violinplot(
                        data=data_fig,
                        color=colors_methods[method],
                        edgecolor='k',
                        cut=0,
                        ax=brick
                    )
                    brick.set(xticklabels=[])
                    brick.set_title(method)
                    brick.set_xlabel("")
                    brick.set_ylabel(m_title)
                    pw_cols.append(brick)
                else:
                    brick = pw.Brick(figsize=(1.68, 2))
                    brick.axis('off')
                    pw_cols.append(brick)
            pw_rows.append(pw.stack(pw_cols, operator="|"))
        pw_fig = pw.stack(pw_rows, operator="/")
        pw_fig.savefig(f"{path}/methods_{m_title}.png", bbox_inches='tight', dpi=200)
        pw_fig.savefig(f"{path}/methods_{m_title}.pdf", bbox_inches='tight')
        pw.clear()


def plot_pyod_outs_regression_error(df, pyod_methods, title, path, thld_outs, col_pred, col_real, col_error):
    # Plot Error distribution in inliers and outliers
    thld = round(len(pyod_methods) * thld_outs)
    df_fig = df.loc[:, [col_real, col_pred, col_error, 'Detections']].copy()
    df_fig.loc[df_fig['Detections'] == 0, 'Type'] = 'Inlier'
    df_fig.loc[df_fig['Detections'] >= thld, 'Type'] = 'Outlier'
    df_fig = df_fig.loc[df_fig['Type'].isin(['Inlier', 'Outlier']), :]
    mae_dict = {
        'Inlier': mean_absolute_error(
            df_fig.loc[df_fig['Type'] == 'Inlier', col_real].values,
            df_fig.loc[df_fig['Type'] == 'Inlier', col_pred].values
        ),
        'Outlier': mean_absolute_error(
            df_fig.loc[df_fig['Type'] == 'Outlier', col_real].values,
            df_fig.loc[df_fig['Type'] == 'Outlier', col_pred].values
        ),
    }
    _, mw_pval = mannwhitneyu(
        df_fig.loc[df_fig['Type'] == 'Inlier', col_error].values,
        df_fig.loc[df_fig['Type'] == 'Outlier', col_error].values,
        alternative='two-sided'
    )
    type_description = {
        'Inlier': '0 Detections',
        'Outlier': f'>= {thld} Detections'
    }
    samples_num = {
        'Inlier':  df_fig[df_fig['Type'] == 'Inlier'].shape[0],
        'Outlier': df_fig[df_fig['Type'] == 'Outlier'].shape[0]
    }
    rename_dict = {x: f"{x}\n{type_description[x]}\n({samples_num[x]} samples)\nMAE={mae_dict[x]:0.2f}" for x in mae_dict}
    df_fig['Type'].replace(rename_dict, inplace=True)
    fig = plt.figure(figsize=(4, 3))
    sns.set_theme(style='whitegrid')
    violin = sns.violinplot(
        data=df_fig,
        x='Type',
        y=col_error,
        palette=['dodgerblue', 'crimson'],
        scale='width',
        order=[rename_dict['Inlier'], rename_dict['Outlier']],
        saturation=0.75,
    )
    pval_formatted = [f"{mw_pval:.2e}"]
    annotator = Annotator(
        violin,
        pairs=[(rename_dict['Inlier'], rename_dict['Outlier'])],
        data=df_fig,
        x='Type',
        y=col_error,
        order=[rename_dict['Inlier'], rename_dict['Outlier']],
    )
    annotator.set_custom_annotations(pval_formatted)
    annotator.configure(loc='outside')
    annotator.annotate()
    plt.title(title, y=1.15)
    plt.savefig(f"{path}/error.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/error.pdf", bbox_inches='tight')
    plt.close(fig)
