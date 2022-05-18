import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)
from experiment.logging import log_hyperparameters
from experiment.regression.shap import perform_shap_explanation
from pytorch_lightning.loggers import LightningLoggerBase
import pandas as pd
from src.utils import utils
import xgboost as xgb
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
from experiment.routines import eval_regression_sa
from experiment.routines import eval_loss, save_feature_importance
from typing import List
from catboost import CatBoost
import lightgbm as lgb
from scripts.python.routines.plot.scatter import add_scatter_trace
import statsmodels.formula.api as smf
import wandb
from scripts.python.routines.plot.p_value import add_p_value_annotation
from scipy.stats import mannwhitneyu
from src.datamodules.cross_validation import RepeatedStratifiedKFoldCVSplitter
from tqdm import tqdm
from sklearn.linear_model import ElasticNet
import pickle


log = utils.get_logger(__name__)

def process(config: DictConfig):

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    if 'wandb' in config.logger:
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
    datamodule.perform_split()
    feature_names = datamodule.get_feature_names()
    outcome_name = datamodule.get_outcome_name()
    df = datamodule.get_df()
    ids_tst = datamodule.ids_tst
    if ids_tst is not None:
        is_test = True
    else:
        is_test = False

    cv_splitter = RepeatedStratifiedKFoldCVSplitter(
        datamodule=datamodule,
        is_split=config.cv_is_split,
        n_splits=config.cv_n_splits,
        n_repeats=config.cv_n_repeats,
        random_state=config.seed,
    )

    best = {}
    if config.direction == "min":
        best["optimized_metric"] = np.Inf
    elif config.direction == "max":
        best["optimized_metric"] = 0.0
    cv_progress = {'fold': [], 'optimized_metric':[]}

    for fold_idx, (ids_trn, ids_val) in tqdm(enumerate(cv_splitter.split())):
        datamodule.ids_trn = ids_trn
        datamodule.ids_val = ids_val
        datamodule.refresh_datasets()
        X_trn = df.loc[df.index[ids_trn], feature_names].values
        y_trn = df.loc[df.index[ids_trn], outcome_name].values
        df.loc[df.index[ids_trn], f"fold_{fold_idx:04d}"] = "train"
        X_val = df.loc[df.index[ids_val], feature_names].values
        y_val = df.loc[df.index[ids_val], outcome_name].values
        df.loc[df.index[ids_val], f"fold_{fold_idx:04d}"] = "val"
        if is_test:
            X_tst = df.loc[df.index[ids_tst], feature_names].values
            y_tst = df.loc[df.index[ids_tst], outcome_name].values
            df.loc[df.index[ids_tst], f"fold_{fold_idx:04d}"] = "test"

        if config.model_type == "xgboost":
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

            y_trn_pred = model.predict(dmat_trn)
            y_val_pred = model.predict(dmat_val)
            if is_test:
                y_tst_pred = model.predict(dmat_tst)

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

        elif config.model_type == "catboost":
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
            model.fit(X_trn, y_trn, eval_set=(X_val, y_val), use_best_model=True)
            model.set_feature_names(feature_names)

            y_trn_pred = model.predict(X_trn).astype('float32')
            y_val_pred = model.predict(X_val).astype('float32')
            if is_test:
                y_tst_pred = model.predict(X_tst).astype('float32')

            metrics_train = pd.read_csv(f"catboost_info/learn_error.tsv", delimiter="\t")
            metrics_val = pd.read_csv(f"catboost_info/test_error.tsv", delimiter="\t")
            loss_info = {
                'epoch': metrics_train.iloc[:, 0],
                'train/loss': metrics_train.iloc[:, 1],
                'val/loss': metrics_val.iloc[:, 1]
            }

            def shap_kernel(X):
                y = model.predict(X)
                return y

            feature_importances = pd.DataFrame.from_dict({'feature': model.feature_names_, 'importance': list(model.feature_importances_)})

        elif config.model_type == "lightgbm":
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

            y_trn_pred = model.predict(X_trn, num_iteration=model.best_iteration).astype('float32')
            y_val_pred = model.predict(X_val, num_iteration=model.best_iteration).astype('float32')
            if is_test:
                y_tst_pred = model.predict(X_tst, num_iteration=model.best_iteration).astype('float32')

            loss_info = {
                'epoch': list(range(len(evals_result['train'][config.lightgbm.metric]))),
                'train/loss': evals_result['train'][config.lightgbm.metric],
                'val/loss': evals_result['val'][config.lightgbm.metric]
            }

            def shap_kernel(X):
                y = model.predict(X, num_iteration=model.best_iteration)
                return y

            feature_importances = pd.DataFrame.from_dict({'feature': model.feature_name(), 'importance': list(model.feature_importance())})

        elif config.model_type == "elastic_net":
            model = ElasticNet(
                alpha=config.elastic_net.alpha,
                l1_ratio=config.elastic_net.l1_ratio,
                max_iter=config.elastic_net.max_iter,
                tol=config.elastic_net.tol,
            ).fit(X_trn, y_trn)

            y_trn_pred = model.predict(X_trn).astype('float32')
            y_val_pred = model.predict(X_val).astype('float32')
            if is_test:
                y_tst_pred = model.predict(X_tst).astype('float32')

            loss_info = {
                'epoch': [0],
                'train/loss': [0],
                'val/loss': [0]
            }

            def shap_kernel(X):
                y = model.predict(X)
                return y

            feature_importances = pd.DataFrame.from_dict({'feature': ['Intercept'] + feature_names, 'importance': [model.intercept_] + list(model.coef_)})

        else:
            raise ValueError(f"Model {config.model_type} is not supported")

        metrics_trn = eval_regression_sa(config, y_trn, y_trn_pred, loggers, 'train', is_log=False, is_save=False)
        metrics_val = eval_regression_sa(config, y_val, y_val_pred, loggers, 'val', is_log=False, is_save=False)
        if is_test:
            metrics_tst = eval_regression_sa(config, y_tst, y_tst_pred, loggers, 'test', is_log=False, is_save=False)

        if config.optimized_part == "train":
            metrics_main = metrics_trn
        elif config.optimized_part == "val":
            metrics_main = metrics_val
        elif config.optimized_part == "test":
            metrics_main = metrics_tst
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
            best["model"] = model
            best['loss_info'] = loss_info
            best['shap_kernel'] = shap_kernel
            best['feature_importances'] = feature_importances
            best['fold'] = fold_idx
            best['ids_trn'] = ids_trn
            best['ids_val'] = ids_val
            df.loc[df.index[ids_trn], "Estimation"] = y_trn_pred
            df.loc[df.index[ids_val], "Estimation"] = y_val_pred
            if is_test:
                df.loc[df.index[ids_tst], "Estimation"] = y_tst_pred

        cv_progress['fold'].append(fold_idx)
        cv_progress['optimized_metric'].append(metrics_main.at[config.optimized_metric, config.optimized_part])

    cv_progress_df = pd.DataFrame(cv_progress)
    cv_progress_df.set_index('fold', inplace=True)
    cv_progress_df.to_excel(f"cv_progress.xlsx", index=True)
    cv_ids = df.loc[:, [f"fold_{fold_idx:04d}" for fold_idx in cv_progress['fold']]]
    cv_ids.to_excel(f"cv_ids.xlsx", index=True)
    predictions = df.loc[:, [f"fold_{best['fold']:04d}", outcome_name, "Estimation"]]
    predictions.to_excel(f"predictions.xlsx", index=True)

    datamodule.ids_trn = best['ids_trn']
    datamodule.ids_val = best['ids_val']

    datamodule.plot_split(f"_best_{best['fold']:04d}")

    y_trn = df.loc[df.index[datamodule.ids_trn], outcome_name].values
    y_trn_pred = df.loc[df.index[datamodule.ids_trn], "Estimation"].values
    y_val = df.loc[df.index[datamodule.ids_val], outcome_name].values
    y_val_pred = df.loc[df.index[datamodule.ids_val], "Estimation"].values
    if is_test:
        y_tst = df.loc[df.index[datamodule.ids_tst], outcome_name].values
        y_tst_pred = df.loc[df.index[datamodule.ids_tst], "Estimation"].values

    metrics_trn = eval_regression_sa(config, y_trn, y_trn_pred, loggers, 'train', is_log=True, is_save=True, suffix=f"_best_{best['fold']:04d}")
    metrics_val = eval_regression_sa(config, y_val, y_val_pred, loggers, 'val', is_log=True, is_save=True, suffix=f"_best_{best['fold']:04d}")
    if is_test:
        metrics_tst = eval_regression_sa(config, y_tst, y_tst_pred, loggers, 'test', is_log=True, is_save=True, suffix=f"_best_{best['fold']:04d}")

    if config.optimized_part == "train":
        metrics_main = metrics_trn
    elif config.optimized_part == "val":
        metrics_main = metrics_val
    elif config.optimized_part == "test":
        metrics_main = metrics_tst
    else:
        raise ValueError(f"Unsupported config.optimized_part: {config.optimized_part}")

    if config.model_type == "xgboost":
        best["model"].save_model(f"epoch_{best['model'].best_iteration}_best_{best['fold']:04d}.model")
    elif config.model_type == "catboost":
        best["model"].save_model(f"epoch_{best['model'].best_iteration_}_best_{best['fold']:04d}.model")
    elif config.model_type == "lightgbm":
        best["model"].save_model(f"epoch_{best['model'].best_iteration}_best_{best['fold']:04d}.txt", num_iteration=best['model'].best_iteration)
    elif config.model_type == "elastic_net":
        pickle.dump(best["model"], open(f"elastic_net_best_{best['fold']:04d}.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Model {config.model_type} is not supported")

    save_feature_importance(best['feature_importances'], config.num_top_features)

    formula = f"Estimation ~ {outcome_name}"
    model_linear = smf.ols(formula=formula, data=df.loc[df.index[datamodule.ids_trn], :]).fit()
    df.loc[df.index[datamodule.ids_trn], "Estimation acceleration"] = df.loc[df.index[datamodule.ids_trn], "Estimation"].values - model_linear.predict(df.loc[df.index[datamodule.ids_trn], :])
    df.loc[df.index[datamodule.ids_val], "Estimation acceleration"] = df.loc[df.index[datamodule.ids_val], "Estimation"].values - model_linear.predict(df.loc[df.index[datamodule.ids_val], :])
    if is_test:
        df.loc[df.index[datamodule.ids_tst], "Estimation acceleration"] = df.loc[df.index[datamodule.ids_tst], "Estimation"].values - model_linear.predict(df.loc[df.index[datamodule.ids_tst], :])
    fig = go.Figure()
    add_scatter_trace(fig, df.loc[df.index[datamodule.ids_trn], outcome_name].values, df.loc[df.index[datamodule.ids_trn], "Estimation"].values, f"Train")
    add_scatter_trace(fig, df.loc[df.index[datamodule.ids_trn], outcome_name].values, model_linear.fittedvalues.values, "", "lines")
    add_scatter_trace(fig, df.loc[df.index[datamodule.ids_val], outcome_name].values, df.loc[df.index[datamodule.ids_val], "Estimation"].values, f"Val")
    if is_test:
        add_scatter_trace(fig, df.loc[df.index[datamodule.ids_tst], outcome_name].values, df.loc[df.index[datamodule.ids_tst], "Estimation"].values, f"Test")
    add_layout(fig, outcome_name, f"Estimation", f"")
    fig.update_layout({'colorway': ['blue', 'blue', 'red', 'green']})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(margin=go.layout.Margin(l=90, r=20, b=80, t=65, pad=0))
    save_figure(fig, f"scatter")

    dist_num_bins = 15
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=df.loc[df.index[datamodule.ids_trn], "Estimation acceleration"].values,
            name=f"Train",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor='blue',
            marker=dict(color='blue', line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(df.loc[df.index[datamodule.ids_trn], "Estimation acceleration"].values) / dist_num_bins,
            opacity=0.8
        )
    )
    fig.add_trace(
        go.Violin(
            y=df.loc[df.index[datamodule.ids_val], "Estimation acceleration"].values,
            name=f"Val",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor='red',
            marker=dict(color='red', line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(df.loc[df.index[datamodule.ids_val], "Estimation acceleration"].values) / dist_num_bins,
            opacity=0.8
        )
    )
    if is_test:
        fig.add_trace(
            go.Violin(
                y=df.loc[df.index[datamodule.ids_tst], "Estimation acceleration"].values,
                name=f"Test",
                box_visible=True,
                meanline_visible=True,
                showlegend=True,
                line_color='black',
                fillcolor='green',
                marker=dict(color='green', line=dict(color='black', width=0.3), opacity=0.8),
                points='all',
                bandwidth=np.ptp(df.loc[df.index[datamodule.ids_tst], "Estimation acceleration"].values) / 50,
                opacity=0.8
            )
        )
    add_layout(fig, "", "Estimation acceleration", f"")
    fig.update_layout({'colorway': ['red', 'blue', 'green']})
    stat_01, pval_01 = mannwhitneyu(df.loc[df.index[datamodule.ids_trn], "Estimation acceleration"].values, df.loc[df.index[datamodule.ids_val], "Estimation acceleration"].values, alternative='two-sided')
    if is_test:
        stat_02, pval_02 = mannwhitneyu(df.loc[df.index[datamodule.ids_trn], "Estimation acceleration"].values, df.loc[df.index[datamodule.ids_tst], "Estimation acceleration"].values, alternative='two-sided')
        stat_12, pval_12 = mannwhitneyu(df.loc[df.index[datamodule.ids_val], "Estimation acceleration"].values, df.loc[df.index[datamodule.ids_tst], "Estimation acceleration"].values, alternative='two-sided')
        fig = add_p_value_annotation(fig, {(0, 1): pval_01, (1, 2): pval_12, (0, 2): pval_02})
    else:
        fig = add_p_value_annotation(fig, {(0, 1): pval_01})
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_layout(margin=go.layout.Margin(l=110, r=20, b=50, t=90, pad=0))
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

    if 'wandb' in config.logger:
        wandb.define_metric(f"epoch")
        wandb.define_metric(f"train/loss")
        wandb.define_metric(f"val/loss")
    eval_loss(best['loss_info'], loggers)

    for logger in loggers:
        logger.save()
    if 'wandb' in config.logger:
        wandb.finish()

    if config.is_shap == True:
        shap_data = {
            'model': best["model"],
            'shap_kernel': best['shap_kernel'],
            'df': df,
            'feature_names': feature_names,
            'outcome_name': outcome_name,
            'ids_all': np.arange(df.shape[0]),
            'ids_trn': datamodule.ids_trn,
            'ids_val': datamodule.ids_val,
            'ids_tst': datamodule.ids_tst
        }
        perform_shap_explanation(config, shap_data)

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_main.at[optimized_metric, config.optimized_part]
