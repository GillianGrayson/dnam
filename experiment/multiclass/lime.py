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


def explain_lime(config, expl_data):

    predict_func = expl_data['predict_func']
    df = expl_data['df']
    feature_names = expl_data['feature_names']
    class_names = expl_data['class_names']
    outcome_name = expl_data['outcome_name']

    num_features = config.lime_num_features
    if num_features == 'all':
        num_features = len(feature_names)

    ids_bkgrd = expl_data[f"ids_{config.lime_bkgrd}"]
    indexes_bkgrd = df.index[ids_bkgrd]
    X_bkgrd = df.loc[indexes_bkgrd, feature_names].values

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_bkgrd,
        feature_names=feature_names,
        class_names=class_names,
        verbose=False,
        mode='classification'
    )

    samples_to_plot_mistakes = {}
    samples_to_plot_corrects = {}
    for part in ['trn', 'val', 'tst', 'all']:
        if expl_data[f"ids_{part}"] is not None:
            Path(f"lime/{part}/samples/mistakes").mkdir(parents=True, exist_ok=True)
            Path(f"lime/{part}/samples/corrects").mkdir(parents=True, exist_ok=True)
            ids = expl_data[f"ids_{part}"]
            indexes = df.index[ids]
            y_real = df.loc[indexes, outcome_name].values
            y_pred = df.loc[indexes, "pred"].values
            is_correct_pred = (np.array(y_real) == np.array(y_pred))
            mistakes_ids = np.where(is_correct_pred == False)[0]
            corrects_ids = np.where(is_correct_pred == True)[0]

            num_mistakes = min(len(mistakes_ids), config.num_examples)
            for m_id in mistakes_ids[0:num_mistakes]:
                if indexes[m_id] in samples_to_plot_mistakes:
                    samples_to_plot_mistakes[indexes[m_id]].append(part)
                else:
                    samples_to_plot_mistakes[indexes[m_id]] = [part]

            correct_samples = {x: 0 for x in range(len(class_names))}
            for c_id in corrects_ids:
                if correct_samples[y_real[c_id]] < config.num_examples:
                    if indexes[c_id] in samples_to_plot_corrects:
                        samples_to_plot_corrects[indexes[c_id]].append(part)
                    else:
                        samples_to_plot_corrects[indexes[c_id]] = [part]
                    correct_samples[y_real[c_id]] += 1

    ids_all = expl_data[f"ids_all"]
    indexes_all = df.index[ids_all]
    df_weights = {}
    for cl_id, cl in enumerate(class_names):
        df_weights[cl] = pd.DataFrame(index=df.index, columns=feature_names)
    for ind in tqdm(indexes_all, desc=f'Calculating LIME explanations'):
        X = df.loc[ind, feature_names].values
        y_real = df.at[ind, outcome_name]
        y_pred = df.at[ind, "pred"]
        y_pred_prob = df.loc[indexes_all, [f"pred_prob_{cl_id}" for cl_id, cl in enumerate(class_names)]].values
        y_pred_raw = df.loc[indexes_all, [f"pred_raw_{cl_id}" for cl_id, cl in enumerate(class_names)]].values

        explanation = explainer.explain_instance(
            data_row=X,
            predict_fn=predict_func,
            num_features=num_features,
            labels=class_names,
            top_labels=len(class_names),
        )

        exp_map = explanation.as_map()
        for cl_id, cl in enumerate(class_names):
            for elem in exp_map[cl_id]:
                df_weights[cl].at[ind, feature_names[elem[0]]] = elem[1]

        if ind in samples_to_plot_mistakes:
            for part in samples_to_plot_mistakes[ind]:
                exp_fig = explanation.as_pyplot_figure()
                plt.title(f"{ind}: Real = {y_real:d}, Pred = {y_pred:d}", {'fontsize': 20})
                ind_save = ind.replace('/', '_')
                exp_fig.savefig(f"lime/{part}/samples/mistakes/{ind_save}.pdf", bbox_inches='tight')
                exp_fig.savefig(f"lime/{part}/samples/mistakes/{ind_save}.png", bbox_inches='tight')
                plt.close()

        if ind in samples_to_plot_corrects:
            for part in samples_to_plot_corrects[ind]:
                exp_fig = explanation.as_pyplot_figure()
                plt.title(f"{ind}: Real = {y_real:d}, Pred = {y_pred:d}", {'fontsize': 20})
                ind_save = ind.replace('/', '_')
                exp_fig.savefig(f"lime/{part}/samples/corrects/{ind_save}.pdf", bbox_inches='tight')
                exp_fig.savefig(f"lime/{part}/samples/corrects/{ind_save}.png", bbox_inches='tight')
                plt.close()

    for cl_id, cl in enumerate(class_names):
        df_weights[cl].dropna(axis=1, how='all', inplace=True)
        df_weights[cl] = df_weights[cl].apply(pd.to_numeric, errors='coerce')
        if config.lime_save_weights:
            df_weights[cl].to_excel(f"lime/weights.xlsx", sheet_name=f"{cl}")

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
