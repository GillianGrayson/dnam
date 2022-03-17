import shap
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from src.utils import utils


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
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[m_id],
                base_values=base_value,
                data=features[m_id],
                feature_names=feature_names
            ),
            max_display=30,
            show=False,
        )
        fig = plt.gcf()
        plt.title(f"{indexes[m_id]}: Real = {y_real[m_id]:0.4f}; Estimated = {y_pred[m_id]:0.4f}", {'fontsize': 20})
        fig.set_size_inches(20, 10, forward=True)
        Path(f"{path}/{indexes[m_id]}_{diff:0.4f}").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{path}/{indexes[m_id]}_{diff:0.4f}/waterfall.pdf", bbox_inches='tight')
        fig.savefig(f"{path}/{indexes[m_id]}_{diff:0.4f}/waterfall.png", bbox_inches='tight')
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

    Path(f"shap").mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values=shap_values[raw_data['ids_train'], :],
        features=raw_data['X_train'],
        feature_names=feature_names,
        max_display=30,
        plot_size=(18, 10),
        show=False,
        plot_type="bar"
    )
    plt.savefig(f'shap/bar.png', bbox_inches='tight')
    plt.savefig(f'shap/bar.pdf', bbox_inches='tight')
    plt.close()

    shap.summary_plot(
        shap_values=shap_values[raw_data['ids_train'], :],
        features=raw_data['X_train'],
        feature_names=feature_names,
        max_display=30,
        plot_size=(18, 10),
        plot_type="violin",
        show=False,
    )
    plt.savefig(f"shap/beeswarm.png", bbox_inches='tight')
    plt.savefig(f"shap/beeswarm.pdf", bbox_inches='tight')
    plt.close()

    local_explain(config, raw_data['y_all'], raw_data['y_all_pred'], raw_data['indexes_all'], shap_values, explainer.expected_value, raw_data['X_all'], feature_names, "shap")
