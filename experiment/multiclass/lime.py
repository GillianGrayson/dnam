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
from scripts.python.routines.plot.scatter import add_scatter_trace
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
                path_curr = f"lime/{part}/samples/mistakes/real({class_names[y_real]})_pred({class_names[y_pred]})"
                Path(f"{path_curr}").mkdir(parents=True, exist_ok=True)
                exp_fig = explanation.as_pyplot_figure()
                plt.title(f"{ind}: Real = {class_names[y_real]}, Pred = {class_names[y_pred]}", {'fontsize': 20})
                ind_save = ind.replace('/', '_')
                exp_fig.savefig(f"{path_curr}/{ind_save}.pdf", bbox_inches='tight')
                exp_fig.savefig(f"{path_curr}/{ind_save}.png", bbox_inches='tight')
                plt.close()

        if ind in samples_to_plot_corrects:
            for part in samples_to_plot_corrects[ind]:
                path_curr = f"lime/{part}/samples/corrects/real({class_names[y_real]})_pred({class_names[y_pred]})"
                Path(f"{path_curr}").mkdir(parents=True, exist_ok=True)
                exp_fig = explanation.as_pyplot_figure()
                plt.title(f"{ind}: Real = {class_names[y_real]}, Pred = {class_names[y_pred]}", {'fontsize': 20})
                ind_save = ind.replace('/', '_')
                exp_fig.savefig(f"{path_curr}/{ind_save}.pdf", bbox_inches='tight')
                exp_fig.savefig(f"{path_curr}/{ind_save}.png", bbox_inches='tight')
                plt.close()

    features_common = set(feature_names)
    with pd.ExcelWriter(f"lime/weights.xlsx") as writer:
        for cl_id, cl in enumerate(class_names):
            df_weights[cl].dropna(axis=1, how='all', inplace=True)
            df_weights[cl] = df_weights[cl].apply(pd.to_numeric, errors='coerce')
            features_common.intersection(set(df_weights[cl].columns.values))
            if config.lime_save_weights:
                df_weights[cl].to_excel(writer, sheet_name=f"{cl}")
        features_common = list(features_common)

    for part in ['trn', 'val', 'tst', 'all']:
        if expl_data[f"ids_{part}"] is not None:
            Path(f"lime/{part}/global").mkdir(parents=True, exist_ok=True)
            ids = expl_data[f"ids_{part}"]
            indexes = df.index[ids]
            X = df.loc[indexes, features_common].values

            lime_weights_all = []
            for cl_id, cl in enumerate(class_names):
                Path(f"lime/{part}/global/{cl}").mkdir(parents=True, exist_ok=True)
                lime_weights = df_weights[cl].loc[indexes, features_common].values
                lime_weights_all.append(lime_weights)

                shap.summary_plot(
                    shap_values=lime_weights,
                    features=X,
                    feature_names=features_common,
                    # max_display=config.num_top_features,
                    plot_type="bar",
                    show=False,
                )
                plt.xlabel('Mean |LIME weights|')
                plt.savefig(f'lime/{part}/global/{cl}/bar.png', bbox_inches='tight')
                plt.savefig(f'lime/{part}/global/{cl}/bar.pdf', bbox_inches='tight')
                plt.close()

                shap.summary_plot(
                    shap_values=lime_weights,
                    features=X,
                    feature_names=features_common,
                    # max_display=config.num_top_features,
                    plot_type="violin",
                    show=False,
                )
                plt.xlabel('LIME weights')
                plt.savefig(f"lime/{part}/global/{cl}/beeswarm.png", bbox_inches='tight')
                plt.savefig(f"lime/{part}/global/{cl}/beeswarm.pdf", bbox_inches='tight')
                plt.close()

            shap.summary_plot(
                shap_values=lime_weights_all,
                features=X,
                feature_names=features_common,
                # max_display=config.num_top_features,
                class_names=class_names,
                class_inds=list(range(len(class_names))),
                show=False,
                color=plt.get_cmap("Set1")
            )
            plt.xlabel('Mean |LIME weights|')
            plt.savefig(f'lime/{part}/global/bar.png', bbox_inches='tight')
            plt.savefig(f'lime/{part}/global/bar.pdf', bbox_inches='tight')
            plt.close()

            mean_abs_lime_weights = np.sum([np.mean(np.absolute(lime_weights_all[cl_id]), axis=0) for cl_id, cl in enumerate(class_names)], axis=0)
            order = np.argsort(mean_abs_lime_weights)[::-1]
            features = np.asarray(feature_names)[order]
            features_best = features[0:config.num_top_features]
            for feat_id, feat in enumerate(features_best):
                fig = go.Figure()
                for cl_id, cl in enumerate(class_names):
                    Path(f"lime/{part}/features/{cl}").mkdir(parents=True, exist_ok=True)
                    lime_weights_cl = lime_weights_all[cl_id][:, order[feat_id]]
                    real_values = df.loc[indexes, feat].values
                    add_scatter_trace(fig, real_values, lime_weights_cl, cl)
                add_layout(fig, f"{feat}", f"LIME weights for<br>{feat}", f"")
                fig.update_layout(legend_font_size=20)
                fig.update_layout(legend={'itemsizing': 'constant'})
                fig.update_layout(
                    margin=go.layout.Margin(
                        l=150,
                        r=20,
                        b=80,
                        t=35,
                        pad=0
                    )
                )
                fig.update_layout({'colorway': px.colors.qualitative.Set1})
                save_figure(fig, f"lime/{part}/features/{feat_id}_{feat}_scatter")

                for cl_id, cl in enumerate(class_names):
                    fig = go.Figure()
                    lime_weights_cl = lime_weights_all[cl_id][:, order[feat_id]]
                    real_values = df.loc[indexes, feat].values
                    add_scatter_trace(fig, real_values, lime_weights_cl, "")
                    add_layout(fig, f"{feat}", f"LIME weights for<br>{feat}", f"")
                    fig.update_layout(legend_font_size=20)
                    fig.update_layout(legend={'itemsizing': 'constant'})
                    fig.update_layout(
                        margin=go.layout.Margin(
                            l=150,
                            r=20,
                            b=80,
                            t=20,
                            pad=0
                        )
                    )
                    fig.update_layout({'colorway': ['red']})
                    save_figure(fig, f"lime/{part}/features/{cl}/{feat_id}_{feat}_scatter")

                fig = go.Figure()
                for cl_id, cl in enumerate(class_names):
                    vals = df.loc[(df.index.isin(indexes)) & (df[outcome_name] == cl_id), feat].values
                    fig.add_trace(
                        go.Violin(
                            y=vals,
                            name=f"{cl}",
                            box_visible=True,
                            meanline_visible=True,
                            showlegend=False,
                            line_color='black',
                            fillcolor=px.colors.qualitative.Set1[cl_id],
                            marker=dict(color=px.colors.qualitative.Set1[cl_id], line=dict(color='black', width=0.3), opacity=0.8),
                            points='all',
                            bandwidth=np.ptp(vals) / 25,
                            opacity=0.8
                        )
                    )
                add_layout(fig, "", f"{feat}", f"")
                fig.update_layout({'colorway': px.colors.qualitative.Set1})
                fig.update_layout(legend={'itemsizing': 'constant'})
                fig.update_layout(title_xref='paper')
                fig.update_layout(legend_font_size=20)
                fig.update_layout(
                    margin=go.layout.Margin(
                        l=130,
                        r=20,
                        b=50,
                        t=20,
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
                save_figure(fig, f"lime/{part}/features/{feat_id}_{feat}_violin")
