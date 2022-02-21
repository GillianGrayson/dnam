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
import xgboost as xgb
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.layout import add_layout
from sa.classification.routines import eval_classification, eval_loss
from typing import List
import wandb


log = utils.get_logger(__name__)

def train_xgboost(config: DictConfig):

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

    dmat_train = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
    dmat_val = xgb.DMatrix(X_val, y_val, feature_names=feature_names)
    dmat_test = xgb.DMatrix(X_test, y_test, feature_names=feature_names)

    class_names = list(datamodule.statuses.keys())

    model_params = {
        'num_class': config.output_dim,
        'booster': config.booster,
        'eta': config.learning_rate,
        'max_depth': config.max_depth,
        'gamma': config.gamma,
        'sampling_method': config.sampling_method,
        'subsample': config.subsample,
        'objective': config.objective,
        'verbosity': config.verbosity,
        'eval_metric': config.eval_metric,
    }
    evals_result = {}
    bst = xgb.train(
        params=model_params,
        dtrain=dmat_train,
        evals=[(dmat_train, "train"), (dmat_val, "val")],
        num_boost_round=config.max_epochs,
        early_stopping_rounds=config.patience,
        evals_result=evals_result
    )
    bst.save_model(f"epoch_{bst.best_iteration}.model")

    fi = bst.get_score(importance_type='weight')
    feature_importances = pd.DataFrame.from_dict({'feature': list(fi.keys()), 'importance': list(fi.values())})
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

    y_train_pred_probs = bst.predict(dmat_train)
    y_train_pred = np.argmax(y_train_pred_probs, 1)
    y_val_pred_probs = bst.predict(dmat_val)
    y_val_pred = np.argmax(y_val_pred_probs, 1)
    y_test_pred_probs = bst.predict(dmat_test)
    y_test_pred = np.argmax(y_test_pred_probs, 1)

    eval_classification(config, 'train', class_names, y_train, y_train_pred, y_train_pred_probs, loggers)
    metrics_val = eval_classification(config, 'val', class_names, y_val, y_val_pred, y_val_pred_probs, loggers)
    eval_classification(config, 'test', class_names, y_test, y_test_pred, y_test_pred_probs, loggers)

    loss_info = {
        'epoch': list(range(len(evals_result['train'][config.eval_metric]))),
        'train/loss': evals_result['train'][config.eval_metric],
        'val/loss': evals_result['val'][config.eval_metric]
    }
    eval_loss(loss_info, loggers)

    for logger in loggers:
        logger.save()
    wandb.finish()

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_val.at[optimized_metric, 'val']

