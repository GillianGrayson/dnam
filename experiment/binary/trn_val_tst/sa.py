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
from experiment.routines import eval_classification_sa
from experiment.routines import eval_loss
from typing import List
from catboost import CatBoost
from src.datamodules.cross_validation import RepeatedStratifiedKFoldCVSplitter
import lightgbm as lgb
import wandb
from tqdm import tqdm
import shap


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
    datamodule.perform_split()
    feature_names = datamodule.get_feature_names()
    class_names = datamodule.get_class_names()
    outcome_name = datamodule.get_outcome_name()
    df = datamodule.get_df()
    ids_tst = datamodule.ids_tst
    if ids_tst is not None:
        is_test = True
    else:
        is_test = False

    cv_datamodule = RepeatedStratifiedKFoldCVSplitter(
        data_module=datamodule,
        n_splits=config.cv_n_splits,
        n_repeats=config.cv_n_repeats,
        groups=config.cv_groups,
        random_state=config.seed,
        shuffle=config.is_shuffle
    )

    best = {}
    if config.direction == "min":
        best["optimized_metric"] = np.Inf
    elif config.direction == "max":
        best["optimized_metric"] = 0.0
    cv_progress = {'fold': [], 'optimized_metric': []}

    for fold_idx, (dl_trn, ids_trn, dl_val, ids_val) in tqdm(enumerate(cv_datamodule.split())):
        datamodule.ids_trn = ids_trn
        datamodule.ids_val = ids_val
        X_trn = df.loc[df.index[ids_trn], feature_names].values
        y_trn = df.loc[df.index[ids_trn], outcome_name].values
        df.loc[df.index[ids_trn], f"fold_{fold_idx:04d}"] = "Train"
        X_val = df.loc[df.index[ids_val], feature_names].values
        y_val = df.loc[df.index[ids_val], outcome_name].values
        df.loc[df.index[ids_val], f"fold_{fold_idx:04d}"] = "Val"
        if is_test:
            X_tst = df.loc[df.index[ids_tst], feature_names].values
            y_tst = df.loc[df.index[ids_tst], outcome_name].values
            df.loc[df.index[ids_tst], f"fold_{fold_idx:04d}"] = "Test"

        if config.model_sa == "xgboost":
            model_params = {
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

            dmat_trn = xgb.DMatrix(X_trn, y_trn, feature_names=feature_names)
            dmat_val = xgb.DMatrix(X_val, y_val, feature_names=feature_names)
            if is_test:
                dmat_tst = xgb.DMatrix(X_tst, y_tst, feature_names=feature_names)

            evals_result = {}
            model = xgb.train(
                params=model_params,
                dtrain=dmat_trn,
                evals=[(dmat_trn, "train"), (dmat_val, "val")],
                num_boost_round=config.max_epochs,
                early_stopping_rounds=config.patience,
                evals_result=evals_result
            )

            y_trn_pred_prob = model.predict(dmat_trn)
            y_val_pred_prob = model.predict(dmat_val)
            y_train_pred = np.array([1 if pred > 0.5 else 0 for pred in y_trn_pred_prob])
            y_val_pred = np.array([1 if pred > 0.5 else 0 for pred in y_val_pred_prob])
            if is_test:
                y_tst_pred_prob = model.predict(dmat_tst)
                y_tst_pred = np.array([1 if pred > 0.5 else 0 for pred in y_tst_pred_prob])

            loss_info = {
                'epoch': list(range(len(evals_result['train'][config.xgboost.eval_metric]))),
                'train/loss': evals_result['train'][config.xgboost.eval_metric],
                'val/loss': evals_result['val'][config.xgboost.eval_metric]
            }

            def shap_kernel(X):
                X = xgb.DMatrix(X, feature_names=feature_names)
                y = model.predict(X)
                return y

            fi = model.get_score(importance_type='weight')
            feature_importances = pd.DataFrame.from_dict({'feature': list(fi.keys()), 'importance': list(fi.values())})

        elif config.model_sa == "catboost":
            model_params = {
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
            model.fit(X_trn, y_trn, eval_set=(X_val, y_val))
            model.set_feature_names(feature_names)

            y_trn_pred_prob = model.predict(X_trn, prediction_type="Probability")
            y_val_pred_prob = model.predict(X_val, prediction_type="Probability")
            y_train_pred = np.argmax(y_trn_pred_prob, 1)
            y_val_pred = np.argmax(y_val_pred_prob, 1)
            if is_test:
                y_tst_pred_prob = model.predict(X_tst, prediction_type="Probability")
                y_tst_pred = np.argmax(y_tst_pred_prob, 1)

            metrics_train = pd.read_csv(f"catboost_info/learn_error.tsv", delimiter="\t")
            metrics_test = pd.read_csv(f"catboost_info/test_error.tsv", delimiter="\t")
            loss_info = {
                'epoch': metrics_train.iloc[:, 0],
                'train/loss': metrics_train.iloc[:, 1],
                'val/loss': metrics_test.iloc[:, 1]
            }

            def shap_kernel(X):
                y = model.predict(X)
                return y

            feature_importances = pd.DataFrame.from_dict({'feature': model.feature_names_, 'importance': list(model.feature_importances_)})

        elif config.model_sa == "lightgbm":
            model_params = {
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

            ds_trn = lgb.Dataset(X_trn, label=y_trn, feature_name=feature_names)
            ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_trn, feature_name=feature_names)
            evals_result = {}
            model = lgb.train(
                params=model_params,
                train_set=ds_trn,
                num_boost_round=config.max_epochs,
                valid_sets=[ds_val, ds_trn],
                valid_names=['val', 'train'],
                evals_result=evals_result,
                early_stopping_rounds=config.patience,
                verbose_eval=True
            )

            y_trn_pred_prob = model.predict(X_trn, num_iteration=model.best_iteration)
            y_val_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
            y_train_pred = np.array([1 if pred > 0.5 else 0 for pred in y_trn_pred_prob])
            y_val_pred = np.array([1 if pred > 0.5 else 0 for pred in y_val_pred_prob])
            if is_test:
                y_tst_pred_prob = model.predict(X_tst, num_iteration=model.best_iteration)
                y_tst_pred = np.array([1 if pred > 0.5 else 0 for pred in y_tst_pred_prob])

            loss_info = {
                'epoch': list(range(len(evals_result['train'][config.lightgbm.metric]))),
                'train/loss': evals_result['train'][config.lightgbm.metric],
                'val/loss': evals_result['val'][config.lightgbm.metric]
            }

            def shap_kernel(X):
                y = model.predict(X, num_iteration=model.best_iteration)
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

    eval_classification_sa(config, 'train', class_names, y_train, y_train_pred, y_trn_pred_prob, loggers)
    metrics_val = eval_classification_sa(config, 'val', class_names, y_val, y_val_pred, y_val_pred_prob, loggers)
    eval_classification_sa(config, 'test', class_names, y_test, y_tst_pred, y_tst_pred_prob, loggers)

    wandb.define_metric(f"epoch")
    wandb.define_metric(f"train/loss")
    wandb.define_metric(f"val/loss")
    eval_loss(loss_info, loggers)

    for logger in loggers:
        logger.save()
    wandb.finish()

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_val.at[optimized_metric, 'val']
