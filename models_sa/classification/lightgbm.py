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
from models_sa.classification.routines import eval_classification, eval_loss
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

    log.info("Logging hyperparameters!")
    log_hyperparameters(loggers, config)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()
    feature_names = datamodule.dnam.columns.to_list()
    data = pd.merge(datamodule.pheno.loc[:, datamodule.outcome], datamodule.dnam, left_index=True, right_index=True)
    train_data = data.iloc[datamodule.ids_trn]
    val_data = data.iloc[datamodule.ids_val]
    test_data = data.iloc[datamodule.ids_tst]
    X_train = train_data.loc[:, datamodule.dnam.columns.values].values
    y_train = train_data.loc[:, datamodule.outcome].values
    X_val = val_data.loc[:, datamodule.dnam.columns.values].values
    y_val = val_data.loc[:, datamodule.outcome].values
    X_test = test_data.loc[:, datamodule.dnam.columns.values].values
    y_test = test_data.loc[:, datamodule.outcome].values

    ds_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train, feature_name=feature_names)
    ds_test = lgb.Dataset(X_test, label=y_test, reference=ds_train, feature_name=feature_names)

    class_names = list(datamodule.statuses.keys())

    model_params = {
        'num_class': config.output_dim,
        'objective': config.objective,
        'boosting': config.boosting,
        'learning_rate': config.learning_rate,
        'num_leaves': config.num_leaves,
        'device': config.device,
        'max_depth': config.max_depth,
        'min_data_in_leaf': config.min_data_in_leaf,
        'feature_fraction': config.feature_fraction,
        'bagging_fraction': config.bagging_fraction,
        'bagging_freq': config.bagging_freq,
        'verbose': config.verbose,
        'metric': config.metric
    }
    evals_result = {}  # to record eval results for plotting
    bst = lgb.train(
        params=model_params,
        train_set=ds_train,
        num_boost_round=config.max_epochs,
        valid_sets=[ds_val, ds_train],
        valid_names=['val', 'train'],
        evals_result=evals_result,
        early_stopping_rounds=config.patience,
        verbose_eval=True
    )
    bst.save_model(f"epoch_{bst.best_iteration}.txt", num_iteration=bst.best_iteration)

    feature_importances = pd.DataFrame.from_dict({'feature': bst.feature_name(), 'importance': list(bst.feature_importance())})
    feature_importances.sort_values(['importance'], ascending=[False], inplace=True)
    fig = go.Figure()
    ys = feature_importances['feature'][0:config.num_top_features][::-1]
    xs = feature_importances['importance'][0:config.num_top_features][::-1]
    add_bar_trace(fig, x=xs, y=ys, text=xs, orientation='h')
    add_layout(fig, f"Feature importance", f"", "")
    fig.update_yaxes(tickfont_size=10)
    fig.update_xaxes(showticklabels=True)
    fig.update_layout(margin=go.layout.Margin(l=110, r=20, b=75, t=25, pad=0))
    save_figure(fig, f"feature_importances")
    feature_importances.set_index('feature', inplace=True)
    feature_importances.to_excel("feature_importances.xlsx", index=True)

    y_train_pred_probs = bst.predict(X_train, num_iteration=bst.best_iteration)
    y_train_pred = np.argmax(y_train_pred_probs, 1)
    y_val_pred_probs = bst.predict(X_val, num_iteration=bst.best_iteration)
    y_val_pred = np.argmax(y_val_pred_probs, 1)
    y_test_pred_probs = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_test_pred = np.argmax(y_test_pred_probs, 1)

    eval_classification(config, 'train', class_names, y_train, y_train_pred, y_train_pred_probs)
    metrics_val = eval_classification(config, 'val', class_names, y_val, y_val_pred, y_val_pred_probs)
    eval_classification(config, 'test', class_names, y_test, y_test_pred, y_test_pred_probs)

    loss_info = {
        'epoch': list(range(len(evals_result['train'][config.metric]))),
        'train/loss': evals_result['train'][config.metric],
        'val/loss': evals_result['val'][config.metric]
    }
    eval_loss(loss_info)

    wandb.finish()

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_val.at[optimized_metric, 'val']

