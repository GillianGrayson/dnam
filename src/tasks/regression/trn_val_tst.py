from typing import List, Optional
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    Trainer,
    seed_everything,
)
import statsmodels.formula.api as smf
from pytorch_lightning.loggers import LightningLoggerBase
import plotly.graph_objects as go
import xgboost as xgb
from catboost import CatBoost
import lightgbm
from sklearn.linear_model import ElasticNet
from src.datamodules.cross_validation import RepeatedStratifiedKFoldCVSplitter
from src.datamodules.tabular import TabularDataModule
import numpy as np
from src.utils import utils
import pandas as pd
from tqdm import tqdm
from src.tasks.regression.shap import explain_shap
from src.tasks.regression.lime import explain_lime
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
from src.tasks.routines import eval_regression, eval_loss, save_feature_importance
from datetime import datetime
from pathlib import Path
import pickle
import wandb
import glob
import os
import shap
from src.models.tabular.base import get_model_framework_dict


log = utils.get_logger(__name__)


def trn_val_tst_regression(config: DictConfig) -> Optional[float]:

    if "seed" in config:
        seed_everything(config.seed, workers=True)

    if 'wandb' in config.logger:
        config.logger.wandb["project"] = config.project_name

    model_framework_dict = get_model_framework_dict()
    model_framework = model_framework_dict[config.model.name]

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: TabularDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.perform_split()
    features = datamodule.get_features()
    num_features = len(features['all'])
    config.in_dim = num_features
    target = datamodule.get_target()
    df = datamodule.get_data()
    ids_tst = datamodule.ids_tst

    colors = datamodule.colors

    cv_splitter = RepeatedStratifiedKFoldCVSplitter(
        datamodule=datamodule,
        is_split=config.cv_is_split,
        n_splits=config.cv_n_splits,
        n_repeats=config.cv_n_repeats,
        random_state=config.seed
    )

    best = {}
    if config.direction == "min":
        best["optimized_metric"] = np.Inf
    elif config.direction == "max":
        best["optimized_metric"] = 0.0

    metrics_cv = pd.DataFrame(columns=['fold', 'optimized_metric'])
    feature_importances_cv = pd.DataFrame(columns=['fold'] + features['all'])

    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_name = config.callbacks.model_checkpoint.filename

    for fold_idx, (ids_trn, ids_val) in tqdm(enumerate(cv_splitter.split())):
        datamodule.ids_trn = ids_trn
        datamodule.ids_val = ids_val
        datamodule.refresh_datasets()
        X_trn = df.loc[df.index[ids_trn], features['all']].values
        y_trn = df.loc[df.index[ids_trn], target].values
        df.loc[df.index[ids_trn], f"fold_{fold_idx:04d}"] = "trn"
        X_val = df.loc[df.index[ids_val], features['all']].values
        y_val = df.loc[df.index[ids_val], target].values
        df.loc[df.index[ids_val], f"fold_{fold_idx:04d}"] = "val"
        X_tst = {}
        y_tst = {}
        for tst_set_name in ids_tst:
            X_tst[tst_set_name] = df.loc[df.index[ids_tst[tst_set_name]], features['all']].values
            y_tst[tst_set_name] = df.loc[df.index[ids_tst[tst_set_name]], target].values
            if tst_set_name != 'tst_all':
                df.loc[df.index[ids_tst[tst_set_name]], f"fold_{fold_idx:04d}"] = tst_set_name

        ckpt_curr = ckpt_name + f"_fold_{fold_idx:04d}"
        if 'csv' in config.logger:
            config.logger.csv["version"] = f"fold_{fold_idx}"
        if 'wandb' in config.logger:
            config.logger.wandb["version"] = f"fold_{fold_idx}_{start_time}"

        if model_framework == "pytorch":
            # Init lightning model
            widedeep = datamodule.get_widedeep()
            embedding_dims = [(x[1], x[2]) for x in widedeep['cat_embed_input']] if widedeep['cat_embed_input'] else []
            if config.model.name.startswith('widedeep'):
                config.model.column_idx = widedeep['column_idx']
                config.model.cat_embed_input = widedeep['cat_embed_input']
                config.model.continuous_cols = widedeep['continuous_cols']
            elif config.model.name.startswith('pytorch_tabular'):
                config.model.continuous_cols = features['con']
                config.model.categorical_cols = features['cat']
                config.model.embedding_dims = embedding_dims
            elif config.model.name == 'nam':
                num_unique_vals = [len(np.unique(X_trn[:, i])) for i in range(X_trn.shape[1])]
                num_units = [min(config.model.num_basis_functions, i * config.model.units_multiplier) for i in num_unique_vals]
                config.model.num_units = num_units

            log.info(f"Instantiating model <{config.model._target_}>")
            model = hydra.utils.instantiate(config.model)
            if config.print_model:
                print(model)

            # Init lightning callbacks
            config.callbacks.model_checkpoint.filename = ckpt_curr
            callbacks: List[Callback] = []
            if "callbacks" in config:
                for _, cb_conf in config.callbacks.items():
                    if "_target_" in cb_conf:
                        log.info(f"Instantiating callback <{cb_conf._target_}>")
                        callbacks.append(hydra.utils.instantiate(cb_conf))

        # Init lightning loggers
        loggers: List[LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    loggers.append(hydra.utils.instantiate(lg_conf))

        if model_framework == "pytorch":
            # Init lightning trainer
            log.info(f"Instantiating trainer <{config.trainer._target_}>")
            trainer: Trainer = hydra.utils.instantiate(
                config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
            )
            log.info("Logging hyperparameters!")
            utils.log_hyperparameters_pytorch(
                config=config,
                model=model,
                datamodule=datamodule,
                trainer=trainer,
                callbacks=callbacks,
                logger=loggers,
            )
        elif model_framework == "stand_alone":
            log.info("Logging hyperparameters!")
            utils.log_hyperparameters_stand_alone(
                config=config,
                logger=loggers,
            )
        else:
            raise ValueError(f"Unsupported model_framework: {model_framework}")

        # Train the model
        if model_framework == "pytorch":
            log.info("Starting training!")
            trainer.fit(model=model, datamodule=datamodule)

            # Evaluate model on test set, using the best model achieved during training
            if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
                log.info("Starting testing!")
                tst_dataloaders = datamodule.test_dataloaders()
                if 'tst_all' in tst_dataloaders:
                    tst_dataloader = tst_dataloaders['tst_all']
                else:
                    tst_dataloader = tst_dataloaders[list(tst_dataloaders.keys())[0]]
                if tst_dataloader is not None and len(tst_dataloader) > 0:
                    trainer.test(model, tst_dataloader, ckpt_path="best")
                else:
                    log.info("Test data is empty!")

            datamodule.dataloaders_evaluate = True
            trn_dataloader = datamodule.train_dataloader()
            val_dataloader = datamodule.val_dataloader()
            tst_dataloaders = datamodule.test_dataloaders()
            datamodule.dataloaders_evaluate = False

            y_trn_pred = torch.cat(trainer.predict(model, dataloaders=trn_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy().ravel()
            y_val_pred = torch.cat(trainer.predict(model, dataloaders=val_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy().ravel()
            y_tst_pred = {}
            for tst_set_name in ids_tst:
                y_tst_pred[tst_set_name] = torch.cat(trainer.predict(model, dataloaders=tst_dataloaders[tst_set_name], return_predictions=True, ckpt_path="best")).cpu().detach().numpy().ravel()

            # Feature importance
            if Path(f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt").is_file():
                model = type(model).load_from_checkpoint(
                    checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                model.eval()
                model.freeze()
            feature_importances = model.get_feature_importance(X_trn, features, config.feature_importance)

        elif model_framework == "stand_alone":
            if config.model.name == "xgboost":
                model_params = {
                    'booster': config.model.booster,
                    'eta': config.model.learning_rate,
                    'max_depth': config.model.max_depth,
                    'gamma': config.model.gamma,
                    'sampling_method': config.model.sampling_method,
                    'subsample': config.model.subsample,
                    'objective': config.model.objective,
                    'verbosity': config.model.verbosity,
                    'eval_metric': config.model.eval_metric,
                }

                dmat_trn = xgb.DMatrix(X_trn, y_trn, feature_names=features['all'])
                dmat_val = xgb.DMatrix(X_val, y_val, feature_names=features['all'])
                dmat_tst = {}
                for tst_set_name in ids_tst:
                    dmat_tst[tst_set_name] = xgb.DMatrix(X_tst[tst_set_name], y_tst[tst_set_name], feature_names=features['all'])

                evals_result = {}
                model = xgb.train(
                    params=model_params,
                    dtrain=dmat_trn,
                    evals=[(dmat_trn, "train"), (dmat_val, "val")],
                    num_boost_round=config.max_epochs,
                    early_stopping_rounds=config.patience,
                    evals_result=evals_result,
                    verbose_eval=False
                )

                y_trn_pred = model.predict(dmat_trn)
                y_val_pred = model.predict(dmat_val)
                y_tst_pred = {}
                for tst_set_name in ids_tst:
                    y_tst_pred[tst_set_name] = model.predict(dmat_tst[tst_set_name])

                loss_info = {
                    'epoch': list(range(len(evals_result['train'][config.model.eval_metric]))),
                    'trn/loss': evals_result['train'][config.model.eval_metric],
                    'val/loss': evals_result['val'][config.model.eval_metric]
                }

                if config.feature_importance == "native":
                    fi = model.get_score(importance_type='weight')
                    fi_features = list(fi.keys())
                    fi_importances = list(fi.values())
                elif config.feature_importance.startswith("shap"):
                    if config.feature_importance == "shap_tree":
                        explainer = shap.TreeExplainer(model, data=X_trn, feature_perturbation='interventional')
                        shap_values = explainer.shap_values(X_trn)
                    elif config.feature_importance == "shap_kernel":
                        def predict_func(X):
                            X = xgb.DMatrix(X, feature_names=features['all'])
                            y = model.predict(X)
                            return y
                        explainer = shap.KernelExplainer(predict_func, X_trn)
                        shap_values = explainer.shap_values(X_trn)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                    else:
                        raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                    fi_features = features['all']
                    fi_importances = np.mean(np.abs(shap_values), axis=0)
                elif config.feature_importance == "none":
                    fi_features = features['all']
                    fi_importances = np.zeros(len(features['all']))
                else:
                    raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                feature_importances = pd.DataFrame.from_dict(
                    {
                        'feature': fi_features,
                        'importance': fi_importances
                    }
                )

            elif config.model.name == "catboost":
                model_params = {
                    'loss_function': config.model.loss_function,
                    'learning_rate': config.model.learning_rate,
                    'depth': config.model.depth,
                    'min_data_in_leaf': config.model.min_data_in_leaf,
                    'max_leaves': config.model.max_leaves,
                    'task_type': config.model.task_type,
                    'verbose': config.model.verbose,
                    'iterations': config.model.max_epochs,
                    'early_stopping_rounds': config.model.patience
                }

                model = CatBoost(params=model_params)
                model.fit(X_trn, y_trn, eval_set=(X_val, y_val), use_best_model=True)
                model.set_feature_names(features['all'])

                y_trn_pred = model.predict(X_trn).astype('float32')
                y_val_pred = model.predict(X_val).astype('float32')
                y_tst_pred = {}
                for tst_set_name in ids_tst:
                    y_tst_pred[tst_set_name] = model.predict(X_tst[tst_set_name]).astype('float32')

                metrics_train = pd.read_csv(f"catboost_info/learn_error.tsv", delimiter="\t")
                metrics_val = pd.read_csv(f"catboost_info/test_error.tsv", delimiter="\t")
                loss_info = {
                    'epoch': metrics_train.iloc[:, 0],
                    'trn/loss': metrics_train.iloc[:, 1],
                    'val/loss': metrics_val.iloc[:, 1]
                }

                if config.feature_importance == "native":
                    fi_features = model.feature_names_
                    fi_importances = list(model.feature_importances_)
                elif config.feature_importance.startswith("shap"):
                    if config.feature_importance == "shap_tree":
                        explainer = shap.TreeExplainer(model, data=X_trn, feature_perturbation='interventional')
                        shap_values = explainer.shap_values(X_trn)
                    elif config.feature_importance == "shap_kernel":
                        def predict_func(X):
                            y = model.predict(X)
                            return y
                        explainer = shap.KernelExplainer(predict_func, X_trn)
                        shap_values = explainer.shap_values(X_trn)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                    else:
                        raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                    fi_features = features['all']
                    fi_importances = np.mean(np.abs(shap_values), axis=0)
                elif config.feature_importance == "none":
                    fi_features = features['all']
                    fi_importances = np.zeros(len(features['all']))
                else:
                    raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                feature_importances = pd.DataFrame.from_dict(
                    {
                        'feature': fi_features,
                        'importance': fi_importances
                    }
                )

            elif config.model.name == "lightgbm":
                model_params = {
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
                    'verbose': config.model.verbose,
                    'metric': config.model.metric,
                }

                ds_trn = lightgbm.Dataset(X_trn, label=y_trn, feature_name=features['all'])
                ds_val = lightgbm.Dataset(X_val, label=y_val, reference=ds_trn, feature_name=features['all'])

                evals_result = {}
                model = lightgbm.train(
                    params=model_params,
                    train_set=ds_trn,
                    num_boost_round=config.max_epochs,
                    valid_sets=[ds_val, ds_trn],
                    valid_names=['val', 'train'],
                    evals_result=evals_result,
                    early_stopping_rounds=config.patience,
                    verbose_eval=False
                )

                y_trn_pred = model.predict(X_trn, num_iteration=model.best_iteration).astype('float32')
                y_val_pred = model.predict(X_val, num_iteration=model.best_iteration).astype('float32')
                y_tst_pred = {}
                for tst_set_name in ids_tst:
                    y_tst_pred[tst_set_name] = model.predict(X_tst[tst_set_name], num_iteration=model.best_iteration).astype('float32')

                loss_info = {
                    'epoch': list(range(len(evals_result['train'][config.model.metric]))),
                    'trn/loss': evals_result['train'][config.model.metric],
                    'val/loss': evals_result['val'][config.model.metric]
                }

                if config.feature_importance == "native":
                    fi_features = model.feature_name()
                    fi_importances = list(model.feature_importance())
                elif config.feature_importance.startswith("shap"):
                    if config.feature_importance == "shap_tree":
                        explainer = shap.TreeExplainer(model, data=X_trn, feature_perturbation='interventional')
                        shap_values = explainer.shap_values(X_trn)
                    elif config.feature_importance == "shap_kernel":
                        def predict_func(X):
                            y = model.predict(X, num_iteration=model.best_iteration)
                            return y
                        explainer = shap.KernelExplainer(predict_func, X_trn)
                        shap_values = explainer.shap_values(X_trn)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                    else:
                        raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                    fi_features = features['all']
                    fi_importances = np.mean(np.abs(shap_values), axis=0)
                elif config.feature_importance == "none":
                    fi_features = features['all']
                    fi_importances = np.zeros(len(features['all']))
                else:
                    raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                feature_importances = pd.DataFrame.from_dict(
                    {
                        'feature': fi_features,
                        'importance': fi_importances
                    }
                )

            elif config.model.name == "elastic_net":
                model = ElasticNet(
                    alpha=config.model.alpha,
                    l1_ratio=config.model.l1_ratio,
                    max_iter=config.model.max_iter,
                    tol=config.model.tol,
                ).fit(X_trn, y_trn)

                y_trn_pred = model.predict(X_trn).astype('float32')
                y_val_pred = model.predict(X_val).astype('float32')
                y_tst_pred = {}
                for tst_set_name in ids_tst:
                    y_tst_pred[tst_set_name] = model.predict(X_tst[tst_set_name]).astype('float32')

                loss_info = {
                    'epoch': [0],
                    'trn/loss': [0],
                    'val/loss': [0]
                }

                if config.feature_importance == "native":
                    fi_features = ['Intercept'] + features['all']
                    fi_importances = [model.intercept_] + list(model.coef_)
                elif config.feature_importance.startswith("shap"):
                    if config.feature_importance == "shap_tree":
                        explainer = shap.TreeExplainer(model, data=X_trn, feature_perturbation='interventional')
                        shap_values = explainer.shap_values(X_trn)
                    elif config.feature_importance == "shap_kernel":
                        def predict_func(X):
                            y = model.predict(X)
                            return y
                        explainer = shap.KernelExplainer(predict_func, X_trn)
                        shap_values = explainer.shap_values(X_trn)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                    else:
                        raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                    fi_features = features['all']
                    fi_importances = np.mean(np.abs(shap_values), axis=0)
                elif config.feature_importance == "none":
                    fi_features = features['all']
                    fi_importances = np.zeros(len(features['all']))
                else:
                    raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                feature_importances = pd.DataFrame.from_dict(
                    {
                        'feature': fi_features,
                        'importance': fi_importances
                    }
                )

            else:
                raise ValueError(f"Model {config.model.name} is not supported")

        else:
            raise ValueError(f"Unsupported model_framework: {model_framework}")

        metrics_trn = eval_regression(config, y_trn, y_trn_pred, loggers, 'trn', is_log=True, is_save=False)
        for m in metrics_trn.index.values:
            metrics_cv.at[fold_idx, f"trn_{m}"] = metrics_trn.at[m, 'trn']
        metrics_val = eval_regression(config, y_val, y_val_pred, loggers, 'val', is_log=True, is_save=False)
        for m in metrics_val.index.values:
            metrics_cv.at[fold_idx, f"val_{m}"] = metrics_val.at[m, 'val']
        metrics_tst = {}
        for tst_set_name in ids_tst:
            metrics_tst[tst_set_name] = eval_regression(config, y_tst[tst_set_name], y_tst_pred[tst_set_name], loggers, tst_set_name, is_log=True, is_save=False)
            for m in metrics_tst[tst_set_name].index.values:
                metrics_cv.at[fold_idx, f"{tst_set_name}_{m}"] = metrics_tst[tst_set_name].at[m, tst_set_name]

        # Make sure everything closed properly
        if model_framework == "pytorch":
            log.info("Finalizing!")
            utils.finish(
                config=config,
                model=model,
                datamodule=datamodule,
                trainer=trainer,
                callbacks=callbacks,
                logger=loggers,
            )
        elif model_framework == "stand_alone":
            if 'wandb' in config.logger:
                wandb.define_metric(f"epoch")
                wandb.define_metric(f"trn/loss")
                wandb.define_metric(f"val/loss")
            eval_loss(loss_info, loggers, is_log=True, is_save=False)
            for logger in loggers:
                logger.save()
            if 'wandb' in config.logger:
                wandb.finish()
        else:
            raise ValueError(f"Unsupported model_framework: {model_framework}")

        if config.optimized_part == "trn":
            metrics_main = metrics_trn
        elif config.optimized_part == "val":
            metrics_main = metrics_val
        elif config.optimized_part.startwith("tst"):
            metrics_main = metrics_tst[config.optimized_part]
        else:
            raise ValueError(f"Unsupported config.optimized_part: {config.optimized_part}")

        if config.direction == "min":
            if metrics_main.at[config.optimized_metric, config.optimized_part] < best["optimized_metric"]:
                is_renew = True
            else:
                is_renew = False
        elif config.direction == "max":
            if metrics_main.at[config.optimized_metric, config.optimized_part] > best["optimized_metric"]:
                is_renew = True
            else:
                is_renew = False

        if is_renew:
            best["optimized_metric"] = metrics_main.at[config.optimized_metric, config.optimized_part]
            if model_framework == "pytorch":
                if Path(f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt").is_file():
                    model = type(model).load_from_checkpoint(checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                    model.eval()
                    model.freeze()
                best["model"] = model

                def predict_func(X):
                    batch = {
                        'all': torch.from_numpy(np.float32(X[:, features['all_ids']])),
                        'continuous': torch.from_numpy(np.float32(X[:, features['con_ids']])),
                        'categorical': torch.from_numpy(np.float32(X[:, features['cat_ids']])),
                    }
                    tmp = best["model"](batch)
                    return tmp.cpu().detach().numpy()

            elif model_framework == "stand_alone":
                best["model"] = model
                best['loss_info'] = loss_info

                if config.model.name == "xgboost":
                    def predict_func(X):
                        X = xgb.DMatrix(X, feature_names=features['all'])
                        y = best["model"].predict(X)
                        return y
                elif config.model.name == "catboost":
                    def predict_func(X):
                        y = best["model"].predict(X)
                        return y
                elif config.model.name == "lightgbm":
                    def predict_func(X):
                        y = best["model"].predict(X, num_iteration=best["model"].best_iteration)
                        return y
                elif config.model.name == "elastic_net":
                    def predict_func(X):
                        y = best["model"].predict(X)
                        return y
                else:
                    raise ValueError(f"Model {config.model.name} is not supported")

            else:
                raise ValueError(f"Unsupported model_framework: {model_framework}")

            best['predict_func'] = predict_func
            best['feature_importances'] = feature_importances
            best['fold'] = fold_idx
            best['ids_trn'] = ids_trn
            best['ids_val'] = ids_val
            df.loc[df.index[ids_trn], "Estimation"] = y_trn_pred
            df.loc[df.index[ids_val], "Estimation"] = y_val_pred
            for tst_set_name in ids_tst:
                if tst_set_name != 'tst_all':
                    df.loc[df.index[ids_tst[tst_set_name]], "Estimation"] = y_tst_pred[tst_set_name]

        metrics_cv.at[fold_idx, 'fold'] = fold_idx
        metrics_cv.at[fold_idx, 'optimized_metric'] = metrics_main.at[config.optimized_metric, config.optimized_part]
        feature_importances_cv.at[fold_idx, 'fold'] = fold_idx
        for feat in features['all']:
            feature_importances_cv.at[fold_idx, feat] = feature_importances.loc[feature_importances['feature'] == feat, 'importance'].values[0]

    df = df.astype({"Estimation": 'float32'})
    metrics_cv.to_excel(f"metrics_cv.xlsx", index=False)
    feature_importances_cv.to_excel(f"feature_importances_cv.xlsx", index=False)
    cv_ids = df.loc[:, [f"fold_{fold_idx:04d}" for fold_idx in metrics_cv.loc[:, 'fold'].values]]
    cv_ids.to_excel(f"cv_ids.xlsx", index=True)
    predictions = df.loc[:, [f"fold_{best['fold']:04d}", target, "Estimation"]]
    predictions.to_excel(f"predictions.xlsx", index=True)

    datamodule.ids_trn = best['ids_trn']
    datamodule.ids_val = best['ids_val']

    datamodule.plot_split(f"_best_{best['fold']:04d}")

    if model_framework == "pytorch":
        fns = glob.glob(f"{ckpt_name}*.ckpt")
        fns.remove(f"{ckpt_name}_fold_{best['fold']:04d}.ckpt")
        for fn in fns:
            os.remove(fn)
    elif model_framework == "stand_alone":
        eval_loss(best['loss_info'], None, is_log=True, is_save=False, file_suffix=f"_best_{best['fold']:04d}")
        if config.model.name == "xgboost":
            best["model"].save_model(f"epoch_{best['model'].best_iteration}_best_{best['fold']:04d}.model")
        elif config.model.name == "catboost":
            best["model"].save_model(f"epoch_{best['model'].best_iteration_}_best_{best['fold']:04d}.model")
        elif config.model.name == "lightgbm":
            best["model"].save_model(f"epoch_{best['model'].best_iteration}_best_{best['fold']:04d}.model", num_iteration=best['model'].best_iteration)
        elif config.model.name == "elastic_net":
            pickle.dump(best["model"], open(f"elastic_net_best_{best['fold']:04d}.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Model {config.model.name} is not supported")

    y_trn = df.loc[df.index[datamodule.ids_trn], target].values
    y_trn_pred = df.loc[df.index[datamodule.ids_trn], "Estimation"].values
    y_val = df.loc[df.index[datamodule.ids_val], target].values
    y_val_pred = df.loc[df.index[datamodule.ids_val], "Estimation"].values
    y_tst = {}
    y_tst_pred = {}
    for tst_set_name in ids_tst:
        y_tst[tst_set_name] = df.loc[df.index[datamodule.ids_tst[tst_set_name]], target].values
        y_tst_pred[tst_set_name] = df.loc[df.index[datamodule.ids_tst[tst_set_name]], "Estimation"].values

    metrics_trn = eval_regression(config, y_trn, y_trn_pred, None, 'trn', is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
    metrics_names = metrics_trn.index.values
    metrics_trn_cv = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names] + [f"{x}_cv_std" for x in metrics_names], columns=['trn'])
    for metric in metrics_names:
        metrics_trn_cv.at[f"{metric}_cv_mean", 'trn'] = metrics_cv[f"trn_{metric}"].mean()
        metrics_trn_cv.at[f"{metric}_cv_std", 'trn'] = metrics_cv[f"trn_{metric}"].std()
    metrics_trn = pd.concat([metrics_trn, metrics_trn_cv])
    metrics_trn.to_excel(f"metrics_trn_best_{best['fold']:04d}.xlsx", index=True, index_label="metric")

    metrics_val = eval_regression(config, y_val, y_val_pred, None, 'val', is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
    metrics_val_cv = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names] + [f"{x}_cv_std" for x in metrics_names], columns=['val'])
    for metric in metrics_names:
        metrics_val_cv.at[f"{metric}_cv_mean", 'val'] = metrics_cv[f"val_{metric}"].mean()
        metrics_val_cv.at[f"{metric}_cv_std", 'val'] = metrics_cv[f"val_{metric}"].std()
    metrics_val = pd.concat([metrics_val, metrics_val_cv])
    metrics_val.to_excel(f"metrics_val_best_{best['fold']:04d}.xlsx", index=True, index_label="metric")

    metrics_tst = {}
    metrics_tst_cv = {}
    metrics_val_tst_cv_mean = {}
    for tst_set_name in ids_tst:
        metrics_tst[tst_set_name] = eval_regression(config, y_tst[tst_set_name], y_tst_pred[tst_set_name], None, tst_set_name, is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
        metrics_tst_cv[tst_set_name] = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names] + [f"{x}_cv_std" for x in metrics_names], columns=[tst_set_name])
        for metric in metrics_names:
            metrics_tst_cv[tst_set_name].at[f"{metric}_cv_mean", tst_set_name] = metrics_cv[f"{tst_set_name}_{metric}"].mean()
            metrics_tst_cv[tst_set_name].at[f"{metric}_cv_std", tst_set_name] = metrics_cv[f"{tst_set_name}_{metric}"].std()
        metrics_tst[tst_set_name] = pd.concat([metrics_tst[tst_set_name], metrics_tst_cv[tst_set_name]])

        metrics_val_tst_cv_mean[tst_set_name] = pd.DataFrame(index=[f"{x}_cv_mean_val_{tst_set_name}" for x in metrics_names],columns=['val', tst_set_name])
        for metric in metrics_names:
            val_tst_value = 0.5 * (metrics_val.at[f"{metric}_cv_mean", 'val'] + metrics_tst[tst_set_name].at[f"{metric}_cv_mean", tst_set_name])
            metrics_val_tst_cv_mean[tst_set_name].at[f"{metric}_cv_mean_val_{tst_set_name}", 'val'] = val_tst_value
            metrics_val_tst_cv_mean[tst_set_name].at[f"{metric}_cv_mean_val_{tst_set_name}", tst_set_name] = val_tst_value
        metrics_val = pd.concat([metrics_val, metrics_val_tst_cv_mean[tst_set_name].loc[:, ['val']]])
        metrics_tst[tst_set_name] = pd.concat([metrics_tst[tst_set_name], metrics_val_tst_cv_mean[tst_set_name].loc[:, [tst_set_name]]])
        metrics_val.to_excel(f"metrics_val_best_{best['fold']:04d}.xlsx", index=True, index_label="metric")
        metrics_tst[tst_set_name].to_excel(f"metrics_{tst_set_name}_best_{best['fold']:04d}.xlsx", index=True, index_label="metric")

    if config.optimized_part == "trn":
        metrics_main = metrics_trn
    elif config.optimized_part == "val":
        metrics_main = metrics_val
    elif config.optimized_part.startwith("tst"):
        metrics_main = metrics_tst[config.optimized_part]
    else:
        raise ValueError(f"Unsupported config.optimized_part: {config.optimized_part}")

    features_labels = []
    for f in best['feature_importances']['feature'].values:
        features_labels.append(features['labels'][f])
    best['feature_importances']['feature_label'] = features_labels
    save_feature_importance(best['feature_importances'], config.num_top_features)

    formula = f"Estimation ~ {target}"
    model_linear = smf.ols(formula=formula, data=df.loc[df.index[datamodule.ids_trn], :]).fit()
    df.loc[df.index[datamodule.ids_trn], "Estimation acceleration"] = df.loc[df.index[datamodule.ids_trn], "Estimation"].values - model_linear.predict(df.loc[df.index[datamodule.ids_trn], :])
    df.loc[df.index[datamodule.ids_val], "Estimation acceleration"] = df.loc[df.index[datamodule.ids_val], "Estimation"].values - model_linear.predict(df.loc[df.index[datamodule.ids_val], :])
    for tst_set_name in ids_tst:
        df.loc[df.index[datamodule.ids_tst[tst_set_name]], "Estimation acceleration"] = df.loc[df.index[datamodule.ids_tst[tst_set_name]], "Estimation"].values - model_linear.predict(df.loc[df.index[datamodule.ids_tst[tst_set_name]], :])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.loc[df.index[datamodule.ids_trn], target].values,
            y=df.loc[df.index[datamodule.ids_trn], "Estimation"].values,
            showlegend=True,
            name=f"trn",
            mode="markers",
            line_color=colors['trn'],
            marker_color=colors['trn'],
            marker=dict(
                size=8,
                opacity=0.75,
                line=dict(
                    color="black",
                    width=0.5
                )
            )
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[df.index[datamodule.ids_trn], target].values,
            y=model_linear.fittedvalues.values,
            showlegend=False,
            name=f"",
            mode="lines",
            line_color=colors['trn'],
            marker_color=colors['trn'],
            marker=dict(
                size=8,
                opacity=0.75,
                line=dict(
                    color="black",
                    width=0.5
                )
            )
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[df.index[datamodule.ids_val], target].values,
            y=df.loc[df.index[datamodule.ids_val], "Estimation"].values,
            showlegend=True,
            name=f"val",
            mode="markers",
            line_color=colors['val'],
            marker_color=colors['val'],
            marker=dict(
                size=8,
                opacity=0.75,
                line=dict(
                    color="black",
                    width=0.5
                )
            )
        )
    )
    for tst_set_name in ids_tst:
        if tst_set_name != 'tst_all':
            fig.add_trace(
                go.Scatter(
                    x=df.loc[df.index[datamodule.ids_tst[tst_set_name]], target].values,
                    y=df.loc[df.index[datamodule.ids_tst[tst_set_name]], "Estimation"].values,
                    showlegend=True,
                    name=f"{tst_set_name}",
                    mode="markers",
                    line_color=colors[tst_set_name],
                    marker_color=colors[tst_set_name],
                    marker=dict(
                        size=8,
                        opacity=0.75,
                        line=dict(
                            color="black",
                            width=0.5
                        )
                    )
                )
            )
    add_layout(fig, target, f"Estimation", f"")
    fig.update_layout({'colorway': list(colors.values())})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(margin=go.layout.Margin(l=90, r=20, b=80, t=65, pad=0))
    save_figure(fig, f"scatter")

    dist_num_bins = 20
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=df.loc[df.index[datamodule.ids_trn], "Estimation acceleration"].values,
            name=f"trn",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor=colors['trn'],
            marker=dict(color=colors['trn'], line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(df.loc[df.index[datamodule.ids_trn], "Estimation acceleration"].values) / dist_num_bins,
            opacity=0.8
        )
    )
    fig.add_trace(
        go.Violin(
            y=df.loc[df.index[datamodule.ids_val], "Estimation acceleration"].values,
            name=f"val",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor=colors['val'],
            marker=dict(color=colors['val'], line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(df.loc[df.index[datamodule.ids_val], "Estimation acceleration"].values) / dist_num_bins,
            opacity=0.8
        )
    )
    for tst_set_name in ids_tst:
        if tst_set_name != 'tst_all':
            fig.add_trace(
                go.Violin(
                    y=df.loc[df.index[datamodule.ids_tst[tst_set_name]], "Estimation acceleration"].values,
                    name=tst_set_name,
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=True,
                    line_color='black',
                    fillcolor=colors[tst_set_name],
                    marker=dict(color=colors[tst_set_name], line=dict(color='black', width=0.3), opacity=0.8),
                    points='all',
                    bandwidth=np.ptp(df.loc[df.index[datamodule.ids_tst[tst_set_name]], "Estimation acceleration"].values) / dist_num_bins,
                    opacity=0.8
                )
            )
    add_layout(fig, "", "Estimation acceleration", f"")
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_layout(
        margin=go.layout.Margin(
            l=110,
            r=20,
            b=50,
            t=90,
            pad=0
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.25,
            xanchor="center",
            x=0.5
        )
    )
    save_figure(fig, f"violin")

    expl_data = {
        'model': best["model"],
        'predict_func': best['predict_func'],
        'df': df,
        'feature_names': features['all'],
        'target_name': target,
        'ids': {
            'all': np.arange(df.shape[0]),
            'trn': datamodule.ids_trn,
            'val': datamodule.ids_val,
        }
    }
    for tst_set_name in ids_tst:
        expl_data['ids'][tst_set_name] = ids_tst[tst_set_name]

    if config.is_lime == True:
        explain_lime(config, expl_data)
    if config.is_shap == True:
        explain_shap(config, expl_data)

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    optimized_mean = config.get("optimized_mean")
    if optimized_metric:
        if optimized_mean == "":
            return metrics_main.at[optimized_metric, config.optimized_part]
        else:
            return metrics_main.at[f"{optimized_metric}_{optimized_mean}", config.optimized_part]
