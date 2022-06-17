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
    Path(f"{path}").mkdir(parents=True, exist_ok=True)
    diff_y = np.array(y_pred) - np.array(y_real)
    order = np.argsort(diff_y)
    order_abs = np.argsort(np.abs(diff_y))
    num_examples = config.num_examples

    ids = list(set(np.concatenate((order[0:num_examples], order[-num_examples:], order_abs[0:num_examples]))))
    log.info(f"Number of samples: {len(ids)}")
    for m_id in ids:
        diff = diff_y[m_id]
        log.info(f"Plotting sample {m_id}: {indexes[m_id]} (real = {y_real[m_id]:0.4f}, estimated = {y_pred[m_id]:0.4f}) with diff = {diff:0.4f}")

        ind_save = indexes[m_id].replace('/', '_')
        Path(f"{path}/{ind_save}_{diff:0.4f}").mkdir(parents=True, exist_ok=True)

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
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/waterfall.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/waterfall.png", bbox_inches='tight')
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
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/decision.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/decision.png", bbox_inches='tight')
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
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/force.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{ind_save}_{diff:0.4f}/force.png", bbox_inches='tight')
        plt.close()


def perform_shap_explanation(config, shap_data):
    for part in ['trn', 'val', 'tst', 'all']:
        if shap_data[f"ids_{part}"] is not None:
            Path(f"shap/global/{part}").mkdir(parents=True, exist_ok=True)
            model = shap_data['model']
            shap_kernel = shap_data['shap_kernel']
            df = shap_data['df']
            feature_names = shap_data['feature_names']
            outcome_name = shap_data['outcome_name']
            ids = shap_data[f"ids_{part}"]
            indexes = df.index[ids]
            X = df.loc[indexes, feature_names].values
            y_pred = df.loc[indexes, "Estimation"].values

            if config.shap_explainer == "Tree":
                #explainer = shap.TreeExplainer(model)
                explainer = shap.explainers.Tree(model, data=X, feature_perturbation='interventional')
                #explainer = shap.TreeExplainer(model, data=X, feature_perturbation='interventional')
                shap_values = explainer.shap_values(X)
                expected_value = explainer.expected_value
            elif config.shap_explainer == "Kernel":
                explainer = shap.KernelExplainer(shap_kernel, X)
                shap_values = explainer.shap_values(X)
                shap_values = shap_values[0]
                expected_value = explainer.expected_value[0]
            elif config.shap_explainer == "Deep":
                explainer = shap.DeepExplainer(model, torch.from_numpy(X))
                shap_values = explainer.shap_values(torch.from_numpy(X))
                expected_value = explainer.expected_value
            else:
                raise ValueError(f"Unsupported explainer type: {config.shap_explainer}")

            shap.summary_plot(
                shap_values=shap_values,
                features=X,
                feature_names=feature_names,
                # max_display=config.num_top_features,
                plot_type="bar",
                show=False,
            )
            plt.savefig(f'shap/global/{part}/bar.png', bbox_inches='tight')
            plt.savefig(f'shap/global/{part}/bar.pdf', bbox_inches='tight')
            plt.close()

            shap.summary_plot(
                shap_values=shap_values,
                features=X,
                feature_names=feature_names,
                # max_display=config.num_top_features,
                plot_type="violin",
                show=False,
            )
            plt.savefig(f"shap/global/{part}/beeswarm.png", bbox_inches='tight')
            plt.savefig(f"shap/global/{part}/beeswarm.pdf", bbox_inches='tight')
            plt.close()

            explanation = shap.Explanation(
                values=shap_values,
                base_values=np.array([expected_value] * len(ids)),
                data=X,
                feature_names=feature_names
            )
            shap.plots.heatmap(
                explanation,
                show=False,
                # max_display=config.num_top_features,
                instance_order=explanation.sum(1)
            )
            plt.savefig(f"shap/global/{part}/heatmap.png", bbox_inches='tight')
            plt.savefig(f"shap/global/{part}/heatmap.pdf", bbox_inches='tight')
            plt.close()

            Path(f"shap/features/{part}").mkdir(parents=True, exist_ok=True)
            mean_abs_impact = np.mean(np.abs(shap_values), axis=0)
            features_order = np.argsort(mean_abs_impact)[::-1]
            inds_to_plot = features_order[0:config.num_top_features]
            for feat_id, ind in enumerate(inds_to_plot):
                feat = feature_names[ind]
                shap.dependence_plot(
                    ind=ind,
                    shap_values=shap_values,
                    features=X,
                    feature_names=feature_names,
                    show=False,
                )
                plt.savefig(f"shap/features/{part}/{feat_id}_{feat}.png", bbox_inches='tight')
                plt.savefig(f"shap/features/{part}/{feat_id}_{feat}.pdf", bbox_inches='tight')
                plt.close()

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=X[:, ind],
                        y=shap_values[:, ind],
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
                add_layout(fig, feat, f"SHAP value for<br>{feat}", f"", font_size=20)
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
                save_figure(fig, f"shap/features/{part}/{feat_id}_{feat}_scatter")

            local_explain(
                config,
                df.loc[indexes, outcome_name].values,
                df.loc[indexes, "Estimation"].values,
                indexes,
                shap_values,
                expected_value,
                X,
                feature_names,
                f"shap/local/{part}"
            )
