import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)
from experiment.logging import log_hyperparameters
from pytorch_lightning.loggers import LightningLoggerBase
import pandas as pd
from src.utils import utils
import xgboost as xgb
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.layout import add_layout
from experiment.multiclass.routines import eval_classification
from experiment.multiclass.shap import perform_shap_explanation
from experiment.routines import eval_loss
from typing import List
from catboost import CatBoost
import lightgbm as lgb
import wandb
import shap
import copy


log = utils.get_logger(__name__)

def process(config: DictConfig):

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
    feature_names = datamodule.get_feature_names()
    class_names = datamodule.get_class_names()
    raw_data = datamodule.get_raw_data()
    X_train = raw_data['X_train']
    y_train = raw_data['y_train']
    X_val = raw_data['X_val']
    y_val = raw_data['y_val']
    X_test = raw_data['X_test']
    y_test = raw_data['y_test']

    if config.model_sa == "xgboost":
        model_params = {
            'num_class': config.xgboost.output_dim,
            'booster': config.xgboost.booster,
            'eta': config.xgboost.learning_rate,
            'max_depth': config.xgboost.max_depth,
            'gamma': config.xgboost.gamma,
            'sampling_method': config.xgboost.sampling_method,
            'subsample': config.xgboost.subsample,
            'objective': config.xgboost.objective,
            'verbosity': config.xgboost.verbosity,
            'eval_metric': config.xgboost.eval_metric,
        }

        dmat_train = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
        dmat_val = xgb.DMatrix(X_val, y_val, feature_names=feature_names)
        dmat_test = xgb.DMatrix(X_test, y_test, feature_names=feature_names)

        evals_result = {}
        model = xgb.train(
            params=model_params,
            dtrain=dmat_train,
            evals=[(dmat_train, "train"), (dmat_val, "val")],
            num_boost_round=config.max_epochs,
            early_stopping_rounds=config.patience,
            evals_result=evals_result
        )
        model.save_model(f"epoch_{model.best_iteration}.model")

        y_train_pred_probs = model.predict(dmat_train)
        y_val_pred_probs = model.predict(dmat_val)
        y_test_pred_probs = model.predict(dmat_test)
        y_train_pred = np.argmax(y_train_pred_probs, 1)
        y_val_pred = np.argmax(y_val_pred_probs, 1)
        y_test_pred = np.argmax(y_test_pred_probs, 1)

        loss_info = {
            'epoch': list(range(len(evals_result['train'][config.xgboost.eval_metric]))),
            'train/loss': evals_result['train'][config.xgboost.eval_metric],
            'val/loss': evals_result['val'][config.xgboost.eval_metric]
        }

        def shap_proba(X):
            X = xgb.DMatrix(X, feature_names=feature_names)
            y = model.predict(X)
            return y

        fi = model.get_score(importance_type='weight')
        feature_importances = pd.DataFrame.from_dict({'feature': list(fi.keys()), 'importance': list(fi.values())})
    elif config.model_sa == "catboost":
        model_params = {
            'classes_count': config.catboost.output_dim,
            'loss_function': config.catboost.loss_function,
            'learning_rate': config.catboost.learning_rate,
            'depth': config.catboost.depth,
            'min_data_in_leaf': config.catboost.min_data_in_leaf,
            'max_leaves': config.catboost.max_leaves,
            'task_type': config.catboost.task_type,
            'verbose': config.catboost.verbose,
            'iterations': config.catboost.max_epochs,
            'early_stopping_rounds': config.catboost.patience
        }

        model = CatBoost(params=model_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        model.set_feature_names(feature_names)
        model.save_model(f"epoch_{model.best_iteration_}.model")

        y_train_pred_probs = model.predict(X_train, prediction_type="Probability")
        y_val_pred_probs = model.predict(X_val, prediction_type="Probability")
        y_test_pred_probs = model.predict(X_test, prediction_type="Probability")
        y_train_pred = np.argmax(y_train_pred_probs, 1)
        y_val_pred = np.argmax(y_val_pred_probs, 1)
        y_test_pred = np.argmax(y_test_pred_probs, 1)

        metrics_train = pd.read_csv(f"catboost_info/learn_error.tsv", delimiter="\t")
        metrics_test = pd.read_csv(f"catboost_info/test_error.tsv", delimiter="\t")
        loss_info = {
            'epoch': metrics_train.iloc[:, 0],
            'train/loss': metrics_train.iloc[:, 1],
            'val/loss': metrics_test.iloc[:, 1]
        }

        def shap_proba(X):
            y = model.predict(X, prediction_type="Probability")
            return y

        feature_importances = pd.DataFrame.from_dict({'feature': model.feature_names_, 'importance': list(model.feature_importances_)})
    elif config.model_sa == "lightgbm":
        model_params = {
            'num_class': config.lightgbm.output_dim,
            'objective': config.lightgbm.objective,
            'boosting': config.lightgbm.boosting,
            'learning_rate': config.lightgbm.learning_rate,
            'num_leaves': config.lightgbm.num_leaves,
            'device': config.lightgbm.device,
            'max_depth': config.lightgbm.max_depth,
            'min_data_in_leaf': config.lightgbm.min_data_in_leaf,
            'feature_fraction': config.lightgbm.feature_fraction,
            'bagging_fraction': config.lightgbm.bagging_fraction,
            'bagging_freq': config.lightgbm.bagging_freq,
            'verbose': config.lightgbm.verbose,
            'metric': config.lightgbm.metric
        }

        ds_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train, feature_name=feature_names)

        evals_result = {}
        model = lgb.train(
            params=model_params,
            train_set=ds_train,
            num_boost_round=config.max_epochs,
            valid_sets=[ds_val, ds_train],
            valid_names=['val', 'train'],
            evals_result=evals_result,
            early_stopping_rounds=config.patience,
            verbose_eval=True
        )
        model.save_model(f"epoch_{model.best_iteration}.txt", num_iteration=model.best_iteration)

        y_train_pred_probs = model.predict(X_train, num_iteration=model.best_iteration)
        y_val_pred_probs = model.predict(X_val, num_iteration=model.best_iteration)
        y_test_pred_probs = model.predict(X_test, num_iteration=model.best_iteration)
        y_train_pred = np.argmax(y_train_pred_probs, 1)
        y_val_pred = np.argmax(y_val_pred_probs, 1)
        y_test_pred = np.argmax(y_test_pred_probs, 1)

        loss_info = {
            'epoch': list(range(len(evals_result['train'][config.lightgbm.metric]))),
            'train/loss': evals_result['train'][config.lightgbm.metric],
            'val/loss': evals_result['val'][config.lightgbm.metric]
        }

        def shap_proba(X):
            y = model.predict(X)
            return y

        feature_importances = pd.DataFrame.from_dict({'feature': model.feature_name(), 'importance': list(model.feature_importance())})
    else:
        raise ValueError(f"Model {config.model_sa} is not supported")

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

    eval_classification(config, 'train', class_names, y_train, y_train_pred, y_train_pred_probs, loggers)
    metrics_val = eval_classification(config, 'val', class_names, y_val, y_val_pred, y_val_pred_probs, loggers)
    eval_classification(config, 'test', class_names, y_test, y_test_pred, y_test_pred_probs, loggers)

    wandb.define_metric(f"epoch")
    wandb.define_metric(f"train/loss")
    wandb.define_metric(f"val/loss")
    eval_loss(loss_info, loggers)

    for logger in loggers:
        logger.save()
    wandb.finish()

    if config.is_shap == True:
        X_all = np.concatenate((X_train, X_val, X_test))
        y_all = np.concatenate((y_train, y_val, y_test))
        y_all_pred = np.concatenate((y_train_pred, y_val_pred, y_test_pred))
        y_all_pred_probs = np.concatenate((y_train_pred_probs, y_val_pred_probs, y_test_pred_probs))
        perform_shap_explanation(config, model, shap_proba, X_all, y_all, y_all_pred, y_all_pred_probs, feature_names, class_names)

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_val.at[optimized_metric, 'val']
