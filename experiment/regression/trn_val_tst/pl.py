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
import numpy as np
from src.utils import utils
import pandas as pd
from experiment.routines import plot_confusion_matrix
from experiment.regression.shap import perform_shap_explanation
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
from scipy.stats import mannwhitneyu
from scripts.python.routines.plot.p_value import add_p_value_annotation
from scripts.python.routines.plot.layout import add_layout


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
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
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
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    feature_names = datamodule.get_feature_names()
    raw_data = datamodule.get_raw_data()
    X_train = raw_data['X_train']
    y_train = raw_data['y_train']
    indexes_train = raw_data['indexes_train']
    train_data = raw_data['train_data']
    X_val = raw_data['X_val']
    y_val = raw_data['y_val']
    indexes_val = raw_data['indexes_val']
    val_data = raw_data['val_data']

    if 'X_test' in raw_data:
        X_test = raw_data['X_test']
        y_test = raw_data['y_test']
        indexes_test = raw_data['indexes_test']
        test_data = raw_data['test_data']
        is_test = True
    else:
        is_test = False

    model.eval()
    model.freeze()

    def shap_kernel(X):
        X = torch.from_numpy(X)
        tmp = model(X)
        return tmp.cpu().detach().numpy()

    y_train_pred = model(torch.from_numpy(X_train)).cpu().detach().numpy().flatten()
    train_data['Estimation'] = y_train_pred
    y_val_pred = model(torch.from_numpy(X_val)).cpu().detach().numpy().flatten()
    val_data['Estimation'] = y_val_pred
    if is_test:
        y_test_pred = model(X_test).cpu().detach().numpy().flatten()
        test_data['Estimation'] = y_test_pred

    raw_data['y_train_pred'] = y_train_pred
    raw_data['y_val_pred'] = y_val_pred
    if is_test:
        raw_data['y_test_pred'] = y_test_pred

    if config.model._target_ == "src.models.dnam.tabnet.TabNetModel":
        feature_importances = np.zeros((model.hparams.input_dim))
        M_explain, masks = model.forward_masks(torch.from_numpy(X_train))
        feature_importances += M_explain.sum(dim=0).cpu().detach().numpy()
        feature_importances = feature_importances / np.sum(feature_importances)
        feature_importances_df = pd.DataFrame.from_dict(
            {
                'feature': feature_names,
                'importance': feature_importances
            }
        )
        feature_importances_df.sort_values(['importance'], ascending=[False], inplace=True)
        fig = go.Figure()
        ys = feature_importances_df['feature'][0:config.num_top_features][::-1]
        xs = feature_importances_df['importance'][0:config.num_top_features][::-1]
        add_bar_trace(fig, x=xs, y=ys, text=xs, orientation='h')
        add_layout(fig, f"Feature importance", f"", "")
        fig.update_yaxes(tickfont_size=10)
        fig.update_xaxes(showticklabels=True)
        fig.update_layout(margin=go.layout.Margin(l=130, r=20, b=75, t=25, pad=0))
        save_figure(fig, f"feature_importances")
        feature_importances_df.set_index('feature', inplace=True)
        feature_importances_df.to_excel("feature_importances.xlsx", index=True)

    formula = f"Estimation ~ {datamodule.outcome}"
    model_linear = smf.ols(formula=formula, data=train_data).fit()
    train_data[f"Estimation acceleration"] = train_data[f'Estimation'] - model_linear.predict(train_data)
    val_data[f"Estimation acceleration"] = val_data[f'Estimation'] - model_linear.predict(val_data)
    if is_test:
        test_data[f"Estimation acceleration"] = test_data[f'Estimation'] - model_linear.predict(test_data)
    fig = go.Figure()
    add_scatter_trace(fig, train_data.loc[:, datamodule.outcome].values, train_data.loc[:, f"Estimation"].values, f"Train")
    add_scatter_trace(fig, train_data.loc[:, datamodule.outcome].values, model_linear.fittedvalues.values, "", "lines")
    add_scatter_trace(fig, val_data.loc[:, datamodule.outcome].values, val_data.loc[:, f"Estimation"].values, f"Val")
    if is_test:
        add_scatter_trace(fig, test_data.loc[:, datamodule.outcome].values, test_data.loc[:, f"Estimation"].values, f"Test")
    add_layout(fig, datamodule.outcome, f"Estimation", f"")
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
            y=train_data.loc[:, f"Estimation acceleration"].values,
            name=f"Train",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor='blue',
            marker=dict(color='blue', line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(train_data.loc[:, f"Estimation acceleration"].values) / dist_num_bins,
            opacity=0.8
        )
    )
    fig.add_trace(
        go.Violin(
            y=val_data.loc[:, f"Estimation acceleration"].values,
            name=f"Val",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor='red',
            marker=dict(color='red', line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(val_data.loc[:, f"Estimation acceleration"].values) / dist_num_bins,
            opacity=0.8
        )
    )
    if is_test:
        fig.add_trace(
            go.Violin(
                y=test_data.loc[:, f"Estimation acceleration"].values,
                name=f"Test",
                box_visible=True,
                meanline_visible=True,
                showlegend=True,
                line_color='black',
                fillcolor='green',
                marker=dict(color='green', line=dict(color='black', width=0.3), opacity=0.8),
                points='all',
                bandwidth=np.ptp(test_data.loc[:, f"Estimation acceleration"].values) / 50,
                opacity=0.8
            )
        )
    add_layout(fig, "", "Estimation acceleration", f"")
    fig.update_layout({'colorway': ['red', 'blue', 'green']})
    stat_01, pval_01 = mannwhitneyu(train_data.loc[:, f"Estimation acceleration"].values,
                                    val_data.loc[:, f"Estimation acceleration"].values, alternative='two-sided')
    if is_test:
        stat_02, pval_02 = mannwhitneyu(train_data.loc[:, f"Estimation acceleration"].values,
                                        test_data.loc[:, f"Estimation acceleration"].values, alternative='two-sided')
        stat_12, pval_12 = mannwhitneyu(val_data.loc[:, f"Estimation acceleration"].values,
                                        test_data.loc[:, f"Estimation acceleration"].values, alternative='two-sided')
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
        if is_test:
            X_all = np.concatenate((X_train, X_val, X_test))
            y_all = np.concatenate((y_train, y_val, y_test))
            indexes_all = np.concatenate((indexes_train, indexes_val, indexes_test))
            y_all_pred = np.concatenate((y_train_pred, y_val_pred, y_test_pred))
        else:
            X_all = np.concatenate((X_train, X_val))
            y_all = np.concatenate((y_train, y_val))
            indexes_all = np.concatenate((indexes_train, indexes_val))
            y_all_pred = np.concatenate((y_train_pred, y_val_pred))
        ids_train = np.linspace(0, X_train.shape[0] - 1, X_train.shape[0], dtype=int)
        ids_val = np.linspace(X_train.shape[0], X_train.shape[0] + X_val.shape[0] - 1, X_val.shape[0], dtype=int)
        if is_test:
            ids_test = np.linspace(X_train.shape[0] + X_val.shape[0],
                                   X_train.shape[0] + X_val.shape[0] + X_test.shape[0] - 1, X_test.shape[0], dtype=int)
        raw_data['X_all'] = X_all
        raw_data['y_all'] = y_all
        raw_data['y_all_pred'] = y_all_pred
        raw_data['indexes_all'] = indexes_all
        raw_data['ids_train'] = ids_train
        raw_data['ids_val'] = ids_val
        if is_test:
            raw_data['ids_test'] = ids_test
            raw_data['ids_all'] = np.concatenate((ids_train, ids_val, ids_test))
        else:
            raw_data['ids_all'] = np.concatenate((ids_train, ids_val))
        perform_shap_explanation(config, model, shap_kernel, raw_data, feature_names)

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
