import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)
from models_sa.logging import log_hyperparameters
from pytorch_lightning.loggers import LightningLoggerBase
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
from typing import List
import pathlib
import wandb


log = utils.get_logger(__name__)

def train_lightgbm(config: DictConfig):

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    config.logger.wandb["project"] = config.project_name
    # Init lightning loggers
    loggers: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))

    num_top_features = 20

    model_params = {
        'num_class': config.model.output_dim,
        'objective': config.model.objective,
        'boosting': config.model.boosting,
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

    log.info("Logging hyperparameters!")
    log_hyperparameters(loggers, config)

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
    evo_metrics = list(evals_result[parts[0]].keys())
    epochs = np.linspace(1, len(evals_result[parts[0]][evo_metrics[0]]), len(evals_result[parts[0]][evo_metrics[0]]))
    for m in evo_metrics:
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

    y_train_pred_probs = bst.predict(X_train, num_iteration=bst.best_iteration)
    y_train_pred = np.argmax(y_train_pred_probs, 1)
    y_val_pred_probs = bst.predict(X_val, num_iteration=bst.best_iteration)
    y_val_pred = np.argmax(y_val_pred_probs, 1)
    y_test_pred_probs = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_test_pred = np.argmax(y_test_pred_probs, 1)

    metrics_classes_dict = get_metrics_dict(config.model.output_dim, object)
    metrics_summary = {
        'accuracy_macro': 'max',
        'accuracy_weighted': 'max',
        'f1_macro': 'max',
        'cohen_kappa': 'max',
        'matthews_corrcoef': 'max',
        'f1_weighted': 'max',
        'auroc_weighted': 'max',
        'auroc_macro': 'max',
    }
    metrics = [metrics_classes_dict[m]() for m in metrics_summary]
    for p in parts:
        for m, sum in metrics_summary.items():
            wandb.define_metric(f"{p}/{m}", summary=sum)
        for m in evo_metrics:
            wandb.define_metric(f"{p}/{m}", summary='min')

    metrics_dict = {'metric': [m._name for m in metrics]}
    for p in parts:
        if p == 'train':
            y_real = y_train
            y_pred = y_train_pred
            y_pred_probs = y_train_pred_probs
        elif p == 'val':
            y_real = y_val
            y_pred = y_val_pred
            y_pred_probs = y_val_pred_probs
        else:
            y_real = y_test
            y_pred = y_test_pred
            y_pred_probs = y_test_pred_probs
        metrics_dict[p] = []
        log_dict = {}
        for m in metrics:
            if m._name in ['auroc_weighted', 'auroc_macro']:
                m_val = m(y_real, y_pred_probs)
            else:
                m_val = m(y_real, y_pred)
            metrics_dict[p].append(m_val)
            log_dict[f"{p}/{m._name}"] = m_val
        for m in evo_metrics:
            m_val = evals_result[p][m]
            log_dict[f"{p}/{m}"] = m_val
        wandb.log(log_dict)

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
        save_figure(fig, f"confusion_matrix_{p}")
    wandb.finish()

    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    metrics_df.set_index('metric', inplace=True)
    metrics_df.to_excel("metrics.xlsx", index=True)

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_df.at[optimized_metric, 'val']

