from typing import List, Optional
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
import statsmodels.formula.api as smf
from pytorch_lightning.loggers import LightningLoggerBase
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.layout import add_layout
from src.datamodules.cross_validation import RepeatedStratifiedKFoldCVSplitter
import numpy as np
from src.utils import utils
import pandas as pd
from tqdm import tqdm
from experiment.routines import plot_confusion_matrix
from experiment.regression.shap import perform_shap_explanation
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
from scipy.stats import mannwhitneyu
from scripts.python.routines.plot.p_value import add_p_value_annotation
from scripts.python.routines.plot.layout import add_layout
from experiment.routines import eval_regression_sa
from datetime import datetime


log = utils.get_logger(__name__)


def process(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    config.logger.wandb["project"] = config.project_name

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()
    datamodule.perform_split()
    feature_names = datamodule.get_feature_names()
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
    cv_progress = {'fold': [], 'optimized_metric':[]}

    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for fold_idx, (dl_trn, ids_trn, dl_val, ids_val) in tqdm(enumerate(cv_datamodule.split())):
        datamodule.ids_trn = ids_trn
        datamodule.ids_val = ids_val
        datamodule.refresh_datasets()
        df.loc[df.index[ids_trn], f"fold_{fold_idx:04d}"] = "train"
        df.loc[df.index[ids_val], f"fold_{fold_idx:04d}"] = "val"
        if is_test:
            df.loc[df.index[ids_tst], f"fold_{fold_idx:04d}"] = "test"

        if 'csv' in config.logger:
            config.logger.csv["version"] = f"fold_{fold_idx}"
        if 'wandb' in config.logger:
            config.logger.wandb["version"] = f"fold_{fold_idx}_{start_time}"

        # Init lightning model
        log.info(f"Instantiating model <{config.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(config.model)

        # Init lightning callbacks
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

        # Init lightning trainer
        log.info(f"Instantiating trainer <{config.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
        )

        # Send some parameters from config to all lightning loggers
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=loggers,
        )

        # Train the model
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

        # Evaluate model on test set, using the best model achieved during training
        if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            test_dataloader = datamodule.test_dataloader()
            if test_dataloader is not None and len(test_dataloader) > 0:
                trainer.test(model, test_dataloader)
            else:
                log.info("Test data is empty!")

        # Make sure everything closed properly
        log.info("Finalizing!")
        utils.finish(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=loggers,
        )

        X_trn = df.loc[df.index[ids_trn], feature_names].values
        y_trn = df.loc[df.index[ids_trn], outcome_name].values
        X_val = df.loc[df.index[ids_val], feature_names].values
        y_val = df.loc[df.index[ids_val], outcome_name].values
        if is_test:
            X_tst = df.loc[df.index[ids_tst], feature_names].values
            y_tst = df.loc[df.index[ids_tst], outcome_name].values

        model.eval()
        model.freeze()

        feature_importances_raw = np.zeros((len(feature_names)))
        M_explain, masks = model.forward_masks(torch.from_numpy(X_trn))
        feature_importances_raw += M_explain.sum(dim=0).cpu().detach().numpy()
        feature_importances_raw = feature_importances_raw / np.sum(feature_importances_raw)
        feature_importances = pd.DataFrame.from_dict(
            {
                'feature': feature_names,
                'importance': feature_importances_raw
            }
        )

        def shap_kernel(X):
            X = torch.from_numpy(X)
            tmp = model(X)
            return tmp.cpu().detach().numpy()

        y_trn_pred = model(torch.from_numpy(X_trn)).cpu().detach().numpy().flatten()
        y_val_pred = model(torch.from_numpy(X_val)).cpu().detach().numpy().flatten()
        if is_test:
            y_tst_pred = model(X_tst).cpu().detach().numpy().flatten()

        eval_regression_sa(config, y_trn, y_trn_pred, loggers, 'train', is_log=False, is_save=False)
        metrics_val = eval_regression_sa(config, y_val, y_val_pred, loggers, 'val', is_log=False, is_save=False)
        if is_test:
            eval_regression_sa(config, y_tst, y_tst_pred, loggers, 'test', is_log=False, is_save=False)

        if config.direction == "min":
            if metrics_val.at[config.optimized_metric, 'val'] < best["optimized_metric"]:
                is_renew = True
            else:
                is_renew = False
        elif config.direction == "max":
            if metrics_val.at[config.optimized_metric, 'val'] > best["optimized_metric"]:
                is_renew = True
            else:
                is_renew = False

        if is_renew:
            best["optimized_metric"] = metrics_val.at[config.optimized_metric, 'val']
            best["model"] = model
            best["trainer"] = trainer
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
        cv_progress['optimized_metric'].append(metrics_val.at[config.optimized_metric, 'val'])

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

    eval_regression_sa(config, y_trn, y_trn_pred, loggers, 'train', is_log=False, is_save=True)
    metrics_val = eval_regression_sa(config, y_val, y_val_pred, loggers, 'val', is_log=False, is_save=True)
    if is_test:
        eval_regression_sa(config, y_tst, y_tst_pred, loggers, 'test', is_log=False, is_save=True)

    best["trainer"].save_checkpoint(f"best_{best['fold']:04d}.ckpt")

    if best['feature_importances'] is not None:
        feature_importances = best['feature_importances']
        feature_importances.sort_values(['importance'], ascending=[False], inplace=True)
        fig = go.Figure()
        ys = feature_importances['feature'][0:config.num_top_features][::-1]
        xs = feature_importances['importance'][0:config.num_top_features][::-1]
        add_bar_trace(fig, x=xs, y=ys, text=xs, orientation='h')
        add_layout(fig, f"Feature importance", f"", "")
        fig.update_yaxes(tickfont_size=10)
        fig.update_xaxes(showticklabels=True)
        fig.update_layout(margin=go.layout.Margin(l=130, r=20, b=75, t=25, pad=0))
        save_figure(fig, f"feature_importances")
        feature_importances.set_index('feature', inplace=True)
        feature_importances.to_excel("feature_importances.xlsx", index=True)

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
    fig.update_layout(
        margin=go.layout.Margin(
            l=90,
            r=20,
            b=80,
            t=65,
            pad=0
        )
    )
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
    stat_01, pval_01 = mannwhitneyu(
        df.loc[df.index[datamodule.ids_trn], "Estimation acceleration"].values,
        df.loc[df.index[datamodule.ids_val], "Estimation acceleration"].values,
        alternative='two-sided'
    )
    if is_test:
        stat_02, pval_02 = mannwhitneyu(
            df.loc[df.index[datamodule.ids_trn], "Estimation acceleration"].values,
            df.loc[df.index[datamodule.ids_tst], "Estimation acceleration"].values,
            alternative='two-sided'
        )
        stat_12, pval_12 = mannwhitneyu(
            df.loc[df.index[datamodule.ids_val], "Estimation acceleration"].values,
            df.loc[df.index[datamodule.ids_tst], "Estimation acceleration"].values,
            alternative='two-sided'
        )
        fig = add_p_value_annotation(fig, {(0, 1): pval_01, (1, 2): pval_12, (0, 2): pval_02})
    else:
        fig = add_p_value_annotation(fig, {(0, 1): pval_01})
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

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_val.at[optimized_metric, 'val']
