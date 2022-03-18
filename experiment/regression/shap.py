import shap
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from src.utils import utils
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
import plotly.express as px


log = utils.get_logger(__name__)


def local_explain(config, y_real, y_pred, indexes, shap_values, base_value, features, feature_names, path):
    diff_y = np.array(y_pred) - np.array(y_real)
    order = np.argsort(diff_y)
    order_abs = np.argsort(np.abs(diff_y))
    num_examples = config.num_examples

    ids = np.concatenate((order[0:num_examples], order[-num_examples:], order_abs[0:num_examples]))

    for m_id in ids:
        diff = diff_y[m_id]
        log.info(f"Plotting sample {indexes[m_id]} (real = {y_real[m_id]}, estimated = {y_pred[m_id]}) with diff = {diff}")

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[m_id],
                base_values=base_value,
                data=features[m_id],
                feature_names=feature_names
            ),
            # max_display=config.num_top_features,
            show=False,
        )
        fig = plt.gcf()
        plt.title(f"{indexes[m_id]}: Real = {y_real[m_id]:0.4f}, Estimated = {y_pred[m_id]:0.4f}", {'fontsize': 20})
        Path(f"{path}/{indexes[m_id]}_{diff:0.4f}").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{path}/{indexes[m_id]}_{diff:0.4f}/waterfall.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{indexes[m_id]}_{diff:0.4f}/waterfall.png", bbox_inches='tight')
        plt.close()

        shap.plots.decision(
            base_value=base_value,
            shap_values=shap_values[m_id],
            features=features[m_id],
            feature_names=feature_names,
            show=False,
        )
        fig = plt.gcf()
        plt.title(f"{indexes[m_id]}: Real = {y_real[m_id]:0.4f}, Estimated = {y_pred[m_id]:0.4f}", {'fontsize': 20})
        Path(f"{path}/{indexes[m_id]}_{diff:0.4f}").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{path}/{indexes[m_id]}_{diff:0.4f}/decision.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{indexes[m_id]}_{diff:0.4f}/decision.png", bbox_inches='tight')
        plt.close()

        shap.plots.force(
            base_value=base_value,
            shap_values=shap_values[m_id],
            features=features[m_id],
            feature_names=feature_names,
            show=False,
            matplotlib=True
        )
        fig = plt.gcf()
        Path(f"{path}/{indexes[m_id]}_{diff:0.4f}").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{path}/{indexes[m_id]}_{diff:0.4f}/force.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{indexes[m_id]}_{diff:0.4f}/force.png", bbox_inches='tight')
        plt.close()


def perform_shap_explanation(config, model, shap_kernel, raw_data, feature_names):
    if config.shap_explainer == "Tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(raw_data['X_all'])
    elif config.shap_explainer == "Kernel":
        explainer = shap.KernelExplainer(shap_kernel, raw_data['X_train'])
        shap_values = explainer.shap_values(raw_data['X_all'])
    elif config.shap_explainer == "Deep":
        model.produce_probabilities = True
        explainer = shap.DeepExplainer(model, torch.from_numpy(raw_data['X_train']))
        shap_values = explainer.shap_values(torch.from_numpy(raw_data['X_all']))
    else:
        raise ValueError(f"Unsupported explainer type: {config.shap_explainer}")

    Path(f"shap/global").mkdir(parents=True, exist_ok=True)
    Path(f"shap/local").mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values=shap_values[raw_data['ids_train'], :],
        features=raw_data['X_train'],
        feature_names=feature_names,
        # max_display=config.num_top_features,
        show=False,
        plot_type="bar"
    )
    plt.savefig(f'shap/global/bar.png', bbox_inches='tight')
    plt.savefig(f'shap/global/bar.pdf', bbox_inches='tight')
    plt.close()

    shap.summary_plot(
        shap_values=shap_values[raw_data['ids_train'], :],
        features=raw_data['X_train'],
        feature_names=feature_names,
        # max_display=config.num_top_features,
        plot_type="violin",
        show=False,
    )
    plt.savefig(f"shap/global/beeswarm.png", bbox_inches='tight')
    plt.savefig(f"shap/global/beeswarm.pdf", bbox_inches='tight')
    plt.close()

    for part in ['train', 'val', 'test', 'all']:
        if f"ids_{part}" in raw_data:
            explanation = shap.Explanation(
                values=shap_values[raw_data[f'ids_{part}'], :],
                base_values=np.array([explainer.expected_value] * len(raw_data[f'ids_{part}'])),
                data=raw_data[f'X_{part}'],
                feature_names=feature_names
            )
            shap.plots.heatmap(
                explanation,
                show=False,
                # max_display=config.num_top_features,
                instance_order=explanation.sum(1)
            )
            plt.savefig(f"shap/global/heatmap_{part}.png", bbox_inches='tight')
            plt.savefig(f"shap/global/heatmap_{part}.pdf", bbox_inches='tight')
            plt.close()

            Path(f"shap/features/{part}").mkdir(parents=True, exist_ok=True)
            shap_values_part = shap_values[raw_data[f'ids_{part}'], :]
            mean_abs_impact = np.mean(np.abs(shap_values_part), axis=0)
            features_order = np.argsort(mean_abs_impact)[::-1]
            inds_to_plot = features_order[0:config.num_top_features]
            for ind in inds_to_plot:
                feat = feature_names[ind]
                shap.dependence_plot(
                    ind=ind,
                    shap_values=shap_values_part,
                    features=raw_data[f'X_{part}'],
                    feature_names=feature_names,
                    show=False,
                )
                plt.savefig(f"shap/features/{part}/{feat}.png", bbox_inches='tight')
                plt.savefig(f"shap/features/{part}/{feat}.pdf", bbox_inches='tight')
                plt.close()

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=raw_data[f'X_{part}'][:, ind],
                        y=shap_values_part[:, ind],
                        showlegend=False,
                        name=feat,
                        mode='markers',
                        marker=dict(
                            size=10,
                            opacity=0.7,
                            line=dict(
                                width=1
                            ),
                            color=raw_data[f'y_{part}_pred'],
                            colorscale=px.colors.sequential.Bluered,
                            showscale=True,
                            colorbar=dict(title=dict(text="Estimation", font=dict(size=20)), tickfont=dict(size=20))
                        )
                    )
                )
                add_layout(fig, feat, f"SHAP value for<br>{feat}", f"", font_size=20)
                fig.update_layout({'colorway': ['blue', 'blue', 'red', 'green']})
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
                save_figure(fig, f"shap/features/{part}/{feat}_estimation")

    local_explain(config, raw_data['y_all'], raw_data['y_all_pred'], raw_data['indexes_all'], shap_values, explainer.expected_value, raw_data['X_all'], feature_names, "shap/local")
