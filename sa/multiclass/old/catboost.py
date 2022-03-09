import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)
from sa.logging import log_hyperparameters
from pytorch_lightning.loggers import LightningLoggerBase
import pandas as pd
from src.utils import utils
from catboost import CatBoost
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.layout import add_layout
from typing import List
from sa.multiclass.routines import eval_classification, eval_loss
import wandb


log = utils.get_logger(__name__)

def train_catboost(config: DictConfig):

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

    class_names = list(datamodule.statuses.keys())

    model_params = {
        'classes_count': config.output_dim,
        'loss_function': config.loss_function,
        'learning_rate': config.learning_rate,
        'depth': config.depth,
        'min_data_in_leaf': config.min_data_in_leaf,
        'max_leaves': config.max_leaves,
        'task_type': config.task_type,
        'verbose': config.verbose,
        'iterations': config.max_epochs,
        'early_stopping_rounds': config.patience
    }

    bst = CatBoost(params=model_params)
    bst.fit(X_train, y_train, eval_set=(X_val, y_val))
    bst.set_feature_names(feature_names)
    bst.save_model(f"epoch_{bst.best_iteration_}.model")

    feature_importances = pd.DataFrame.from_dict({'feature': bst.feature_names_, 'importance': list(bst.feature_importances_)})
    feature_importances.sort_values(['importance'], ascending=[False], inplace=True)
    ys = feature_importances['feature'][0:config.num_top_features][::-1]
    xs = feature_importances['importance'][0:config.num_top_features][::-1]
    fig = go.Figure()
    add_bar_trace(fig, x=xs, y=ys, text=xs, orientation='h')
    add_layout(fig, f"Feature importance", f"", "")
    fig.update_yaxes(tickfont_size=10)
    fig.update_xaxes(showticklabels=True)
    fig.update_layout(margin=go.layout.Margin(l=110, r=20, b=75, t=25, pad=0))
    save_figure(fig, f"feature_importances")
    feature_importances.set_index('feature', inplace=True)
    feature_importances.to_excel("feature_importances.xlsx", index=True)

    y_train_pred_probs = bst.predict(X_train, prediction_type="Probability")
    y_train_pred = np.argmax(y_train_pred_probs, 1)
    y_val_pred_probs = bst.predict(X_val, prediction_type="Probability")
    y_val_pred = np.argmax(y_val_pred_probs, 1)
    y_test_pred_probs = bst.predict(X_test, prediction_type="Probability")
    y_test_pred = np.argmax(y_test_pred_probs, 1)

    eval_classification(config, 'train', class_names, y_train, y_train_pred, y_train_pred_probs, loggers)
    metrics_val = eval_classification(config, 'val', class_names, y_val, y_val_pred, y_val_pred_probs, loggers)
    eval_classification(config, 'test', class_names, y_test, y_test_pred, y_test_pred_probs, loggers)

    wandb.define_metric(f"epoch")
    wandb.define_metric(f"train/loss")
    wandb.define_metric(f"val/loss")
    metrics_train = pd.read_csv(f"catboost_info/learn_error.tsv", delimiter="\t")
    metrics_test = pd.read_csv(f"catboost_info/test_error.tsv", delimiter="\t")
    loss_info = {
        'epoch': metrics_train.iloc[:, 0],
        'train/loss': metrics_train.iloc[:, 1],
        'val/loss': metrics_test.iloc[:, 1]
    }
    eval_loss(loss_info, loggers)

    for logger in loggers:
        logger.save()
    wandb.finish()

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_val.at[optimized_metric, 'val']

