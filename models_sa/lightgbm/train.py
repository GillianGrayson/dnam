import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)
import matplotlib.pyplot as plt
import pandas as pd
from models_sa.metrics_multiclass import get_metrics_dict
from src.utils import utils
import lightgbm as lgb
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.layout import add_layout
import plotly.express as px
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import pathlib


log = utils.get_logger(__name__)

def train_lightgbm(config: DictConfig):

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    num_top_features = 20

    model_params = {
        'num_class': config.model.output_dim,
        'objective': config.model.objective,
        'boosting': config.model.boosting,
        'num_iterations': config.model.num_iterations,
        'learning_rate': config.model.learning_rate,
        'num_leaves': config.model.num_leaves,
        'device': config.model.device,
        'max_depth': config.model.max_depth,
        'min_data_in_leaf': config.model.min_data_in_leaf,
        'feature_fraction': config.model.feature_fraction,
        'bagging_fraction': config.model.bagging_fraction,
        'bagging_freq': config.model.bagging_freq,
        'verbose': config.model.verbose
    }

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()
    feature_names = datamodule.betas.columns.to_list()
    data = pd.merge(datamodule.pheno.loc[:, datamodule.outcome], datamodule.betas, left_index=True, right_index=True)
    train_data = data.iloc[datamodule.ids_train]
    val_data = data.iloc[datamodule.ids_val]
    test_data = data.iloc[datamodule.ids_test]
    X_train = train_data.loc[:, datamodule.betas.columns.values].values
    y_train = train_data.loc[:, datamodule.outcome].values
    X_val = val_data.loc[:, datamodule.betas.columns.values].values
    y_val = val_data.loc[:, datamodule.outcome].values
    X_test = test_data.loc[:, datamodule.betas.columns.values].values
    y_test = test_data.loc[:, datamodule.outcome].values
    ds_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train, feature_name=feature_names)
    ds_test = lgb.Dataset(X_test, label=y_test, reference=ds_train, feature_name=feature_names)

    max_epochs = config.trainer.max_epochs
    patience = config.trainer.patience

    evals_result = {}  # to record eval results for plotting

    bst = lgb.train(
        params=model_params,
        train_set=ds_train,
        num_boost_round=max_epochs,
        valid_sets=[ds_val, ds_test, ds_train],
        valid_names=['val', 'test', 'train'],
        evals_result=evals_result,
        early_stopping_rounds=patience,
        verbose_eval=True
    )

    bst.save_model(f"epoch_{bst.best_iteration}.txt", num_iteration=bst.best_iteration)

    parts = list(evals_result.keys())
    metrics = list(evals_result[parts[0]].keys())
    epochs = np.linspace(1, len(evals_result[parts[0]][metrics[0]]), len(evals_result[parts[0]][metrics[0]]))
    for m in metrics:
        fig = go.Figure()
        m_dict = {'Epochs': epochs}
        for p in parts:
            ys = evals_result[p][m]
            add_scatter_trace(fig, epochs, ys, f"{p}", mode='lines')
            m_dict[p] = ys
        add_layout(fig, f"Epochs", f"{m}", "")
        fig.update_layout({'colorway': px.colors.qualitative.Set1})
        save_figure(fig, f"{m}")
        m_df = pd.DataFrame.from_dict(m_dict)
        m_df.set_index('Epochs', inplace=True)
        m_df.to_excel(f"{m}.xlsx", index=True)

    feature_importances = pd.DataFrame.from_dict({'feature': bst.feature_name(), 'importance': list(bst.feature_importance())})
    feature_importances.sort_values(['importance'], ascending=[False], inplace=True)
    fig = go.Figure()
    ys = feature_importances['feature'][0:num_top_features][::-1]
    xs = feature_importances['importance'][0:num_top_features][::-1]
    add_bar_trace(fig, x=xs, y=ys, text=xs, orientation='h')
    add_layout(fig, f"Feature importance", f"", "")
    fig.update_yaxes(tickfont_size=10)
    fig.update_xaxes(showticklabels=True)
    save_figure(fig, f"feature_importances")
    feature_importances.set_index('feature', inplace=True)
    feature_importances.to_excel("feature_importances.xlsx", index=True)

    metrics_dict = get_metrics_dict(config.model.output_dim, object)
    metrics = [
        metrics_dict["accuracy_macro"](),
        metrics_dict["accuracy_weighted"](),
        metrics_dict["f1_macro"](),
        metrics_dict["cohen_kappa"](),
        metrics_dict["matthews_corrcoef"](),
        metrics_dict["f1_weighted"](),
    ]

    y_train_pred_probs = bst.predict(X_train, num_iteration=bst.best_iteration)
    y_train_pred = np.argmax(y_train_pred_probs, 1)
    y_val_pred_probs = bst.predict(X_val, num_iteration=bst.best_iteration)
    y_val_pred = np.argmax(y_val_pred_probs, 1)
    y_test_pred_probs = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_test_pred = np.argmax(y_test_pred_probs, 1)

    metrics_dict = {'metric': [m._name for m in metrics]}
    for part in ['train', 'val', 'test']:
        if part == 'train':
            y_real = y_train
            y_pred = y_train_pred
        elif part == 'val':
            y_real = y_val
            y_pred = y_val_pred
        else:
            y_real = y_test
            y_pred = y_test_pred
        metrics_dict[part] = []
        for m in metrics:
            metrics_dict[part].append(m(y_real, y_pred))

        conf_mtx = confusion_matrix(y_real, y_pred)
        statuses = list(datamodule.statuses.keys())
        fig = ff.create_annotated_heatmap(conf_mtx, x=statuses, y=statuses, colorscale='Viridis')
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=-0.45,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        fig.update_layout(margin=dict(t=50, l=200))
        fig['data'][0]['showscale'] = True
        save_figure(fig, f"confusion_matrix_{part}")

    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    metrics_df.set_index('metric', inplace=True)
    metrics_df.to_excel("metrics.xlsx", index=True)

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_df.at[optimized_metric, 'val']

