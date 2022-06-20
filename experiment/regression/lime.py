import pandas as pd
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
from tqdm import tqdm


log = utils.get_logger(__name__)


def explain_lime(config, expl_data):

    predict_func = expl_data['predict_func']
    df = expl_data['df']
    feature_names = expl_data['feature_names']
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
        class_names=[outcome_name],
        verbose=False,
        mode='regression'
    )

    samples_to_plot = {}
    for part in ['trn', 'val', 'tst', 'all']:
        if expl_data[f"ids_{part}"] is not None:
            Path(f"lime/{part}/samples").mkdir(parents=True, exist_ok=True)
            ids = expl_data[f"ids_{part}"]
            indexes = df.index[ids]
            y_real = df.loc[indexes, outcome_name].values
            y_pred = df.loc[indexes, "Estimation"].values
            y_diff = np.array(y_pred) - np.array(y_real)
            order = np.argsort(y_diff)
            order_abs = np.argsort(np.abs(y_diff))
            num_examples = config.num_examples
            samples = set(np.concatenate((order[0:num_examples], order[-num_examples:], order_abs[0:num_examples])))
            for s in samples:
                if s in samples_to_plot:
                    samples_to_plot[s].append(part)
                else:
                    samples_to_plot[s] = [part]

    ids_all = expl_data[f"ids_all"]
    indexes_all = df.index[ids_all]
    df_weights = pd.DataFrame(index=df.index, columns=feature_names)
    for ind in tqdm(indexes_all, desc=f'Calculating LIME explanations'):
        X = df.loc[ind, feature_names].values
        y_real = df.at[ind, outcome_name]
        y_pred = df.at[ind, "Estimation"]
        y_diff = y_pred - y_real

        explanation = explainer.explain_instance(
            data_row=X,
            predict_fn=predict_func,
            num_features=num_features
        )

        exp_map = explanation.as_map()[1]
        for elem in exp_map:
            df_weights.at[ind, feature_names[elem[0]]] = elem[1]

        if ind in samples_to_plot:
            for part in samples_to_plot[ind]:
                exp_fig = explanation.as_pyplot_figure()
                plt.title(f"{ind}: Real = {y_real:0.4f}, Estimated = {y_pred:0.4f}", {'fontsize': 20})
                ind_save = ind.replace('/', '_')
                exp_fig.savefig(f"lime/{part}/samples/{ind_save}_{y_diff:0.4f}.pdf", bbox_inches='tight')
                exp_fig.savefig(f"lime/{part}/samples/{ind_save}_{y_diff:0.4f}.png", bbox_inches='tight')
                plt.close()

    df_weights.dropna(axis=1, how='all', inplace=True)


