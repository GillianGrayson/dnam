import shap
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from src.utils import utils


log = utils.get_logger(__name__)

def global_explain(shap_values, features, feature_names, class_names, path):
    Path(f"{path}").mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values=shap_values,
        features=features,
        feature_names=feature_names,
        max_display=30,
        class_names=class_names,
        class_inds=list(range(len(class_names))),
        plot_size=(18, 10),
        show=False,
        color=plt.get_cmap("Set1")
    )

    plt.savefig(f'{path}/bar.png', bbox_inches='tight')
    plt.savefig(f'{path}/bar.pdf', bbox_inches='tight')
    plt.close()

    for cl_id, cl in enumerate(class_names):
        shap.summary_plot(
            shap_values=shap_values[cl_id],
            features=features,
            feature_names=feature_names,
            max_display=30,
            plot_size=(18, 10),
            plot_type="violin",
            title=cl,
            show=False,
        )
        plt.savefig(f"{path}/beeswarm_{cl}.png", bbox_inches='tight')
        plt.savefig(f"{path}/beeswarm_{cl}.pdf", bbox_inches='tight')
        plt.close()


def local_explain(config, y_real, y_pred, shap_values, base_values, features, feature_names, class_names, path):
    is_correct_pred = (np.array(y_real) == np.array(y_pred))
    mistakes_ids = np.where(is_correct_pred == False)[0]
    num_mistakes = min(len(mistakes_ids), config.num_examples)

    for m_id in mistakes_ids[0:num_mistakes]:
        log.info(f"Plotting sample with error #{m_id}")
        subj_cl = y_real[m_id]
        subj_pred_cl = y_pred[m_id]
        for st_id, st in enumerate(class_names):
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[st_id][m_id],
                    base_values=base_values[st_id],
                    data=features[m_id],
                    feature_names=feature_names
                ),
                max_display=30,
                show=False
            )
            fig = plt.gcf()
            fig.set_size_inches(20, 10, forward=True)
            Path(f"{path}/errors/real({class_names[subj_cl]})_pred({class_names[subj_pred_cl]})/{m_id}").mkdir(
                parents=True, exist_ok=True)
            fig.savefig(
                f"{path}/errors/real({class_names[subj_cl]})_pred({class_names[subj_pred_cl]})/{m_id}/waterfall_{st}.pdf",
                bbox_inches='tight')
            fig.savefig(
                f"{path}/errors/real({class_names[subj_cl]})_pred({class_names[subj_pred_cl]})/{m_id}/waterfall_{st}.png",
                bbox_inches='tight')
            plt.close()

    passed_examples = {x: 0 for x in range(len(class_names))}
    for subj_id in range(features.shape[0]):
        subj_cl = y_real[subj_id]
        if passed_examples[subj_cl] < config.num_examples:
            log.info(f"Plotting correct sample #{passed_examples[subj_cl]} for {subj_cl}")
            for st_id, st in enumerate(class_names):
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[st_id][subj_id],
                        base_values=base_values[st_id],
                        data=features[subj_id],
                        feature_names=feature_names
                    ),
                    max_display=30,
                    show=False
                )
                fig = plt.gcf()
                fig.set_size_inches(20, 10, forward=True)
                Path(f"{path}/corrects/{class_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}").mkdir(parents=True, exist_ok=True)
                fig.savefig(f"{path}/corrects/{class_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}/waterfall_{st}.pdf", bbox_inches='tight')
                fig.savefig(f"{path}/corrects/{class_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}/waterfall_{st}.png", bbox_inches='tight')
                plt.close()
            passed_examples[subj_cl] += 1


def perform_shap_explanation(config, model, shap_proba, raw_data, feature_names, class_names):
    if config.shap_explainer == "Tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(raw_data['X_all'])

        if raw_data['y_all_pred_probs'].shape[1] != len(explainer.expected_value):
            raise ValueError(f"Wrong number of classes in explainer.expected_value")

        shap_values_train = copy.deepcopy(shap_values)
        for cl_id, cl in enumerate(class_names):
            shap_values_train[cl_id] = shap_values[cl_id][raw_data['ids_train'], :]
        X_train = raw_data['X_all'][raw_data['ids_train'], :]

        global_explain(shap_values_train, X_train, feature_names, class_names, "shap/raw")
        local_explain(config, raw_data['y_all'], raw_data['y_all_pred'], shap_values, explainer.expected_value, raw_data['X_all'], feature_names, class_names, "shap/raw")

        # Calculate base prob
        base_prob = []
        base_prob_num = []
        base_prob_den = 0
        for class_id in range(0, len(explainer.expected_value)):
            base_prob_num.append(np.exp(explainer.expected_value[class_id]))
            base_prob_den += np.exp(explainer.expected_value[class_id])
        for class_id in range(0, len(explainer.expected_value)):
            base_prob.append(base_prob_num[class_id] / base_prob_den)

        # Сonvert raw SHAP values to probability SHAP values
        shap_values_prob = copy.deepcopy(shap_values)
        for class_id in range(0, len(explainer.expected_value)):
            for subject_id in range(0, raw_data['y_all_pred_probs'].shape[0]):

                # Сhecking raw SHAP values
                real_raw = raw_data['y_all_pred_raw'][subject_id, class_id]
                expl_raw = explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id])
                diff_raw = real_raw - expl_raw
                if abs(diff_raw) > 1e-6:
                    log.warning(f"Difference between raw for subject {subject_id} in class {class_id}: {abs(diff_raw)}")

                # Checking conversion to probability space
                real_prob = raw_data['y_all_pred_probs'][subject_id, class_id]
                expl_prob_num = np.exp(explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id]))
                expl_prob_den = 0
                for c_id in range(0, len(explainer.expected_value)):
                    expl_prob_den += np.exp(explainer.expected_value[c_id] + sum(shap_values[c_id][subject_id]))
                expl_prob = expl_prob_num / expl_prob_den
                delta_prob = expl_prob - base_prob[class_id]
                diff_prob = real_prob - expl_prob
                if abs(diff_prob) > 1e-6:
                    log.warning(f"Difference between prediction for subject {subject_id} in class {class_id}: {abs(diff_prob)}")

                # Сonvert raw SHAP values to probability SHAP values
                shap_contrib_logodd = np.sum(shap_values[class_id][subject_id])
                shap_contrib_prob = delta_prob
                coeff = shap_contrib_prob / shap_contrib_logodd
                for feature_id in range(0, raw_data['X_all'].shape[1]):
                    shap_values_prob[class_id][subject_id, feature_id] = shap_values[class_id][subject_id, feature_id] * coeff
                diff_check = shap_contrib_prob - sum(shap_values_prob[class_id][subject_id])
                if abs(diff_check) > 1e-6:
                    log.warning(f"Difference between SHAP contribution for subject {subject_id} in class {class_id}: {diff_check}")

        shap_values = shap_values_prob

    elif config.shap_explainer == "Kernel":
        explainer = shap.KernelExplainer(shap_proba, raw_data['X_train'])
        shap_values = explainer.shap_values(raw_data['X_all'])
        base_prob = explainer.expected_value
        for class_id in range(0, raw_data['y_all_pred_probs'].shape[1]):
            for subject_id in range(0, raw_data['y_all_pred_probs'].shape[0]):
                real_prob = raw_data['y_all_pred_probs'][subject_id, class_id]
                expl_prob = explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id])
                diff_prob = real_prob - expl_prob
                if abs(diff_prob) > 1e-6:
                    log.warning(f"Difference between prediction for subject {subject_id} in class {class_id}: {abs(diff_prob)}")

    elif config.shap_explainer == "Deep":
        model.produce_probabilities = True
        explainer = shap.DeepExplainer(model, torch.from_numpy(raw_data['X_train']))
        shap_values = explainer.shap_values(torch.from_numpy(raw_data['X_all']))
        base_prob = explainer.expected_value
        for class_id in range(0, raw_data['y_all_pred_probs'].shape[1]):
            for subject_id in range(0, raw_data['y_all_pred_probs'].shape[0]):
                real_prob = raw_data['y_all_pred_probs'][subject_id, class_id]
                expl_prob = explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id])
                diff_prob = real_prob - expl_prob
                if abs(diff_prob) > 1e-6:
                    log.warning(f"Difference between prediction for subject {subject_id} in class {class_id}: {abs(diff_prob)}")

    else:
        raise ValueError(f"Unsupported explainer type: {config.shap_explainer}")

    shap_values_train = copy.deepcopy(shap_values)
    for cl_id, cl in enumerate(class_names):
        shap_values_train[cl_id] = shap_values[cl_id][raw_data['ids_train'], :]
    X_train = raw_data['X_all'][raw_data['ids_train'], :]

    global_explain(shap_values_train, X_train, feature_names, class_names, "shap/prob")
    local_explain(config, raw_data['y_all'], raw_data['y_all_pred'], shap_values, base_prob, raw_data['X_all'], feature_names, class_names, "shap/prob")
