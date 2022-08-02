import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import shap
from src.utils import utils
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
import plotly.express as px
import lime
import lime.lime_tabular
from tqdm import tqdm


log = utils.get_logger(__name__)

def get_figure_for_sample_explanation(exp_map, feature_names):
    fig = go.Figure()
    xs = [x[1] for x in exp_map][::-1]
    ys = [feature_names[x[0]] for x in exp_map][::-1]
    bases = [0 if x < 0 else 0 for x in xs]
    colors = ['darkblue' if x < 0 else 'crimson' for x in xs]
    fig.add_trace(
        go.Bar(
            x=xs,
            y=list(range(len(ys))),
            orientation='h',
            marker_color=colors,
            base=bases
        )
    )
    add_layout(fig, "LIME weights", "", "")
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
    fig.update_yaxes(tickfont_size=24)
    fig.update_xaxes(tickfont_size=24)
    fig.update_layout(
        autosize=False,
        width=600 + 20 * len(max(ys, key=len)),
        height=40 * len(ys),
        margin=go.layout.Margin(
            l=20 * len(max(ys, key=len)),
            r=20,
            b=100,
            t=60,
            pad=0
        )
    )
    fig.update_layout(legend={'itemsizing': 'constant'})
    return fig


def explain_lime(config, expl_data):

    predict_func = expl_data['predict_func']
    df = expl_data['df']
    feature_names = expl_data['feature_names']
    target_name = expl_data['target_name']

    num_features = config.lime_num_features
    if num_features == 'all':
        num_features = len(feature_names)

    ids_bkgrd = expl_data[f"ids_{config.lime_bkgrd}"]
    indexes_bkgrd = df.index[ids_bkgrd]
    X_bkgrd = df.loc[indexes_bkgrd, feature_names].values

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_bkgrd,
        feature_names=feature_names,
        class_names=[target_name],
        verbose=False,
        mode='regression'
    )

    samples_to_plot = {}
    for part in ['trn', 'val', 'tst', 'all']:
        if expl_data[f"ids_{part}"] is not None:
            Path(f"lime/{part}/samples").mkdir(parents=True, exist_ok=True)
            ids = expl_data[f"ids_{part}"]
            indexes = df.index[ids]
            y_real = df.loc[indexes, target_name].values
            y_pred = df.loc[indexes, "Estimation"].values
            y_diff = np.array(y_pred) - np.array(y_real)
            order = np.argsort(y_diff)
            order_abs = np.argsort(np.abs(y_diff))
            num_examples = config.num_examples
            ids_selected = list(set(np.concatenate((order[0:num_examples], order[-num_examples:], order_abs[0:num_examples]))))
            for s in ids_selected:
                if indexes[s] in samples_to_plot:
                    samples_to_plot[indexes[s]].append(part)
                else:
                    samples_to_plot[indexes[s]] = [part]

    ids_all = expl_data[f"ids_all"]
    indexes_all = df.index[ids_all]
    df_weights = pd.DataFrame(index=df.index, columns=feature_names)
    for ind in tqdm(indexes_all, desc=f'Calculating LIME explanations'):
        X = df.loc[ind, feature_names].values
        y_real = df.at[ind, target_name]
        y_pred = df.at[ind, "Estimation"]
        y_diff = y_pred - y_real

        explanation = explainer.explain_instance(
            data_row=X,
            predict_fn=predict_func,
            num_features=num_features
        )

        if abs(y_pred - explanation.predicted_value) > 1e-5:
            raise ValueError(f"Wrong model prediction for {ind}: predict_func={explanation.predicted_value}, df={y_pred}")

        exp_map = explanation.as_map()[1]
        for elem in exp_map:
            df_weights.at[ind, feature_names[elem[0]]] = elem[1]

        if ind in samples_to_plot:
            for part in samples_to_plot[ind]:
                ind_save = ind.replace('/', '_')
                fig = get_figure_for_sample_explanation(exp_map, feature_names)
                fig.update_layout(
                    title_font_size=25,
                    title_text=f"{ind}: Real: {y_real:0.2f}, Pred: {y_pred:0.2f}, LIME: {explanation.local_pred[0]:0.2f}",
                    title_xanchor="center",
                    title_xref="paper"
                )
                save_figure(fig, f"lime/{part}/samples/{ind_save}_{y_diff:0.4f}")

    df_weights.dropna(axis=1, how='all', inplace=True)
    df_weights = df_weights.apply(pd.to_numeric, errors='coerce')
    if config.lime_save_weights:
        df_weights.to_excel(f"lime/weights.xlsx")

    for part in ['trn', 'val', 'tst', 'all']:
        if expl_data[f"ids_{part}"] is not None:
            Path(f"lime/{part}/global").mkdir(parents=True, exist_ok=True)
            ids = expl_data[f"ids_{part}"]
            indexes = df.index[ids]
            X = df.loc[indexes, df_weights.columns.values].values
            lime_weights = df_weights.loc[indexes, :].values
            y_pred = df.loc[indexes, "Estimation"].values

            shap.summary_plot(
                shap_values=lime_weights,
                features=X,
                feature_names=df_weights.columns.values,
                # max_display=config.num_top_features,
                plot_type="bar",
                show=False,
            )
            plt.xlabel('Mean |LIME weights|')
            plt.savefig(f'lime/{part}/global/bar.png', bbox_inches='tight')
            plt.savefig(f'lime/{part}/global/bar.pdf', bbox_inches='tight')
            plt.close()

            shap.summary_plot(
                shap_values=lime_weights,
                features=X,
                feature_names=df_weights.columns.values,
                # max_display=config.num_top_features,
                plot_type="violin",
                show=False,
            )
            plt.xlabel('LIME weights')
            plt.savefig(f"lime/{part}/global/beeswarm.png", bbox_inches='tight')
            plt.savefig(f"lime/{part}/global/beeswarm.pdf", bbox_inches='tight')
            plt.close()

            Path(f"lime/{part}/features").mkdir(parents=True, exist_ok=True)
            mean_abs_impact = np.mean(np.abs(lime_weights), axis=0)
            features_order = np.argsort(mean_abs_impact)[::-1]
            feat_ids_to_plot = features_order[0:config.num_top_features]
            for rank, feat_id in enumerate(feat_ids_to_plot):
                feat = feature_names[feat_id]
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=X[:, feat_id],
                        y=lime_weights[:, feat_id],
                        showlegend=False,
                        name=feat,
                        mode='markers',
                        marker=dict(
                            size=10,
                            opacity=0.7,
                            line=dict(
                                width=1
                            ),
                            color=y_pred,
                            colorscale=px.colors.sequential.Bluered,
                            showscale=True,
                            colorbar=dict(title=dict(text="Estimation", font=dict(size=20)), tickfont=dict(size=20))
                        )
                    )
                )
                add_layout(fig, feat, f"LIME weights for<br>{feat}", f"", font_size=20)
                fig.update_layout(legend_font_size=20)
                fig.update_layout(legend={'itemsizing': 'constant'})
                fig.update_layout(
                    margin=go.layout.Margin(
                        l=120,
                        r=20,
                        b=80,
                        t=25,
                        pad=0
                    )
                )
                save_figure(fig, f"lime/{part}/features/{rank}_{feat}_scatter")
