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


def plot_pyod_plots(df, pyod_methods, color, title, path):
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
                        linewidth=0.1,
                        ax=brick
                    )
                    brick.set_title(method)
                    brick.set_xlabel(m_title)
                    brick.set_ylabel('Count')
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