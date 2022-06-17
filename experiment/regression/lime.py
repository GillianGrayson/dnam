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
import lime
import lime.lime_tabular


log = utils.get_logger(__name__)


def perform_lime_explanation(config, shap_data):
    if shap_data[f"ids_trn"] is not None:
        predict_func = shap_data['shap_kernel']
        feature_names = shap_data['feature_names']
        outcome_name = shap_data['outcome_name']
        df = shap_data['df']
        ids_trn = shap_data[f"ids_trn"]
        indexes_trn = df.index[ids_trn]
        X_trn = df.loc[indexes_trn, feature_names].values
        for part in ['val', 'tst']:
            if shap_data[f"ids_{part}"] is not None:
                Path(f"lime/{part}").mkdir(parents=True, exist_ok=True)
                ids = shap_data[f"ids_{part}"]
                indexes = df.index[ids]

                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_trn,
                    feature_names=feature_names,
                    class_names=['Estimation'],
                    verbose=True,
                    mode='regression'
                )

                for ind in indexes:
                    X = df.loc[ind, feature_names].values
                    y_pred = df.at[ind, "Estimation"]
                    y_real = df.at[ind, outcome_name]
                    y_diff = y_pred - y_real
                    explanation = explainer.explain_instance(
                        data_row=X,
                        predict_fn=predict_func,
                        num_features=config.num_top_features
                    )

                    exp_fig = explanation.as_pyplot_figure()
                    plt.title(f"{ind}: Real = {y_real:0.4f}, Estimated = {y_pred:0.4f}",{'fontsize': 20})
                    ind_save = ind.replace('/', '_')
                    exp_fig.savefig(f"lime/{part}/{ind_save}_{y_diff:0.4f}.pdf", bbox_inches='tight')
                    exp_fig.savefig(f"lime/{part}/{ind_save}_{y_diff:0.4f}.png", bbox_inches='tight')
                    # explanation.save_to_file(f"lime/{part}/{ind}_{y_diff:0.4f}.html")
                    plt.close()

                    exp_map = explanation.as_map()


