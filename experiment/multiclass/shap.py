import shap
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path


def perform_shap_explanation(config, model, shap_proba, X, y_real, y_pred, y_prob, feature_names, class_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if y_prob.shape[1] != len(explainer.expected_value):
        raise ValueError(f"Wrong number of classes in explainer.expected_value")

    # Calculate base prob
    base_prob = []
    base_prob_num = []
    base_prob_den = 0
    for class_id in range(0, len(explainer.expected_value)):
        base_prob_num.append(np.exp(explainer.expected_value[class_id]))
        base_prob_den += np.exp(explainer.expected_value[class_id])
    for class_id in range(0, len(explainer.expected_value)):
        base_prob.append(base_prob_num[class_id] / base_prob_den)

    # Checking conversion to probability space
    delta_probs = []
    for class_id in range(0, y_prob.shape[1]):
        delta_probs.append([])
        for subject_id in range(0, y_prob.shape[0]):
            real_prob = y_prob[subject_id, class_id]
            expl_prob_num = np.exp(explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id]))
            expl_prob_den = 0
            for c_id in range(0, len(explainer.expected_value)):
                expl_prob_den += np.exp(explainer.expected_value[c_id] + sum(shap_values[c_id][subject_id]))
            expl_prob = expl_prob_num / expl_prob_den
            delta_probs[class_id].append(expl_prob - base_prob[class_id])
            diff_prob = real_prob - expl_prob
            if abs(diff_prob) > 1e-6:
                print(f"Difference between prediction for subject {subject_id} in class {class_id}: {abs(diff_prob)}")

    # Convert logloss SHAP values to probability SHAP values
    shap_values_prob = copy.deepcopy(shap_values)
    for class_id in range(0, len(explainer.expected_value)):
        for subject_id in range(0, y_prob.shape[0]):
            shap_contrib_logloss = np.sum(shap_values[class_id][subject_id])
            shap_contrib_prob = delta_probs[class_id][subject_id]
            if np.sign(shap_contrib_logloss) != np.sign(shap_contrib_prob):
                print(f"Different signs in logloss and probability SHAP contribution for subject {subject_id} in class {class_id}")
            coeff = shap_contrib_prob / shap_contrib_logloss
            for feature_id in range(0, X.shape[1]):
                shap_values_prob[class_id][subject_id, feature_id] = shap_values[class_id][subject_id, feature_id] * coeff
            diff_check = shap_contrib_prob - sum(shap_values_prob[class_id][subject_id])
            if abs(diff_check) > 1e-5:
                print(f"Difference between SHAP contribution for subject {subject_id} in class {class_id}: {diff_check}")
    shap_values = shap_values_prob

    shap.summary_plot(
        shap_values=shap_values,
        features=X,
        feature_names=feature_names,
        max_display=30,
        class_names=class_names,
        class_inds=list(range(len(class_names))),
        plot_size=(18, 10),
        show=False,
        color=plt.get_cmap("Set1")
    )
    Path(f"shap").mkdir(parents=True, exist_ok=True)
    plt.savefig('shap/bar.png')
    plt.savefig('shap/bar.pdf')
    plt.close()

    for st_id, st in enumerate(class_names):
        shap.summary_plot(
            shap_values=shap_values[st_id],
            features=X,
            feature_names=feature_names,
            max_display=30,
            plot_size=(18, 10),
            plot_type="violin",
            title=st,
            show=False
        )
        plt.savefig(f"shap/beeswarm_{st}.png")
        plt.savefig(f"shap/beeswarm_{st}.pdf")
        plt.close()

    is_correct_pred = (np.array(y_real) == np.array(y_pred))
    mistakes_ids = np.where(is_correct_pred == False)[0]
    num_mistakes = min(len(mistakes_ids), config.num_examples)

    for m_id in mistakes_ids[0:num_mistakes]:
        subj_cl = y_real[m_id]
        subj_pred_cl = y_pred[m_id]
        for st_id, st in enumerate(class_names):
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[st_id][m_id],
                    base_values=base_prob[st_id],
                    data=X[m_id],
                    feature_names=feature_names
                ),
                max_display=30,
                show=False
            )
            fig = plt.gcf()
            fig.set_size_inches(18, 10, forward=True)
            Path(f"shap/errors/real({class_names[subj_cl]})_pred({class_names[subj_pred_cl]})/{m_id}").mkdir(parents=True, exist_ok=True)
            fig.savefig(f"shap/errors/real({class_names[subj_cl]})_pred({class_names[subj_pred_cl]})/{m_id}/waterfall_{st}.pdf")
            fig.savefig(f"shap/errors/real({class_names[subj_cl]})_pred({class_names[subj_pred_cl]})/{m_id}/waterfall_{st}.png")
            plt.close()

    passed_examples = {x: 0 for x in range(len(class_names))}
    for subj_id in range(X.shape[0]):
        subj_cl = y_real[subj_id]
        if passed_examples[subj_cl] < config.num_examples:
            for st_id, st in enumerate(class_names):
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[st_id][subj_id],
                        base_values=base_prob[st_id],
                        data=X[subj_id],
                        feature_names=feature_names
                    ),
                    max_display=30,
                    show=False
                )
                fig = plt.gcf()
                fig.set_size_inches(18, 10, forward=True)
                Path(f"shap/corrects/{class_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}").mkdir(parents=True, exist_ok=True)
                fig.savefig(f"shap/corrects//{class_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}/waterfall_{st}.pdf")
                fig.savefig(f"shap/corrects//{class_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}/waterfall_{st}.png")
                plt.close()
            passed_examples[subj_cl] += 1


    #explainer_ker = shap.KernelExplainer(shap_proba, data=X)
    #shap_values_ker = explainer_ker.shap_values(X)