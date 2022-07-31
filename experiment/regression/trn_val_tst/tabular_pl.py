from typing import List, Optional
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningModule,
    Trainer,
    seed_everything,
)
import statsmodels.formula.api as smf
from pytorch_lightning.loggers import LightningLoggerBase
import plotly.graph_objects as go
from src.models.tabular.widedeep.tab_mlp import WDTabMLPModel
from src.models.tabular.widedeep.tab_resnet import WDTabResnetModel
from src.models.tabular.pytorch_tabular.autoint import PTAutoIntModel
from src.models.tabular.pytorch_tabular.tabnet import PTTabNetModel
from src.models.tabular.pytorch_tabular.node import PTNODEModel
from src.models.tabular.pytorch_tabular.category_embedding import PTCategoryEmbeddingModel
from src.datamodules.cross_validation_tabular import RepeatedStratifiedKFoldCVSplitter
from src.datamodules.tabular import TabularDataModule
import numpy as np
from src.utils import utils
import pandas as pd
from tqdm import tqdm
from experiment.regression.shap import explain_shap
from experiment.regression.lime import explain_lime
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.save import save_figure
from scipy.stats import mannwhitneyu
from scripts.python.routines.plot.p_value import add_p_value_annotation
from scripts.python.routines.plot.layout import add_layout
from experiment.routines import eval_regression, save_feature_importance
from datetime import datetime
from pathlib import Path


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

    if 'wandb' in config.logger:
        config.logger.wandb["project"] = config.project_name

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: TabularDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.perform_split()
    feature_names = datamodule.get_feature_names()
    num_features = len(feature_names['all'])
    config.in_dim = num_features
    target_name = datamodule.get_target()
    df = datamodule.get_data()
    ids_tst = datamodule.ids_tst
    if len(ids_tst) > 0:
        is_tst = True
    else:
        is_tst = False

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

    cv_progress = pd.DataFrame(columns=['fold', 'optimized_metric'])

    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_name = config.callbacks.model_checkpoint.filename

    for fold_idx, (ids_trn, ids_val) in tqdm(enumerate(cv_splitter.split())):
        datamodule.ids_trn = ids_trn
        datamodule.ids_val = ids_val
        datamodule.refresh_datasets()
        df.loc[df.index[ids_trn], f"fold_{fold_idx:04d}"] = "trn"
        df.loc[df.index[ids_val], f"fold_{fold_idx:04d}"] = "val"
        if is_tst:
            df.loc[df.index[ids_tst], f"fold_{fold_idx:04d}"] = "tst"


        config.callbacks.model_checkpoint.filename = ckpt_name + f"_fold_{fold_idx:04d}"

        if 'csv' in config.logger:
            config.logger.csv["version"] = f"fold_{fold_idx}"
        if 'wandb' in config.logger:
            config.logger.wandb["version"] = f"fold_{fold_idx}_{start_time}"

        # Init lightning model
        widedeep = datamodule.get_widedeep()
        embedding_dims = [(x[1], x[2]) for x in widedeep['cat_embed_input']] if widedeep['cat_embed_input'] else []
        if config.model_type == "widedeep_tab_mlp":
            config.model = config["widedeep_tab_mlp"]
            config.model.column_idx = widedeep['column_idx']
            config.model.cat_embed_input = widedeep['cat_embed_input']
            config.model.continuous_cols = widedeep['continuous_cols']
        elif config.model_type == "widedeep_tab_resnet":
            config.model = config["widedeep_tab_resnet"]
            config.model.column_idx = widedeep['column_idx']
            config.model.cat_embed_input = widedeep['cat_embed_input']
            config.model.continuous_cols = widedeep['continuous_cols']
        elif config.model_type == "pytorch_tabular_autoint":
            config.model = config["pytorch_tabular_autoint"]
            config.model.continuous_cols = feature_names['con']
            config.model.categorical_cols = feature_names['cat']
        elif config.model_type == "pytorch_tabular_tabnet":
            config.model = config["pytorch_tabular_tabnet"]
            config.model.continuous_cols = feature_names['con']
            config.model.categorical_cols = feature_names['cat']
            config.model.embedding_dims = embedding_dims
        elif config.model_type == "pytorch_tabular_node":
            config.model = config["pytorch_tabular_node"]
            config.model.continuous_cols = feature_names['con']
            config.model.categorical_cols = feature_names['cat']
            config.model.embedding_dims = embedding_dims
        elif config.model_type == "pytorch_tabular_category_embedding":
            config.model = config["pytorch_tabular_category_embedding"]
            config.model.continuous_cols = feature_names['con']
            config.model.categorical_cols = feature_names['cat']
            config.model.embedding_dims = embedding_dims
        else:
            raise ValueError(f"Unsupported model: {config.model_type}")

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
                trainer.test(model, test_dataloader, ckpt_path="best")
            else:
                log.info("Test data is empty!")

        datamodule.dataloaders_evaluate = True
        trn_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        tst_dataloader = datamodule.test_dataloader()
        datamodule.dataloaders_evaluate = False

        y_trn = df.loc[df.index[ids_trn], target_name].values
        y_val = df.loc[df.index[ids_val], target_name].values
        if is_tst:
            y_tst = df.loc[df.index[ids_tst], target_name].values

        y_trn_pred = torch.cat(trainer.predict(model, dataloaders=trn_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy().ravel()
        y_val_pred = torch.cat(trainer.predict(model, dataloaders=val_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy().ravel()
        if is_tst:
            y_tst_pred = torch.cat(trainer.predict(model, dataloaders=tst_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy().ravel()

        if config.model_type == "widedeep_tab_mlp":
            feature_importances = None
        elif config.model_type == "widedeep_tab_resnet":
            feature_importances = None
        elif config.model_type == "pytorch_tabular_autoint":
            feature_importances = None
        elif config.model_type == "pytorch_tabular_tabnet":
            feature_importances = None
        elif config.model_type == "pytorch_tabular_node":
            feature_importances = None
        elif config.model_type == "pytorch_tabular_category_embedding":
            feature_importances = None
        else:
            raise ValueError(f"Unsupported model: {config.model_type}")

        metrics_trn = eval_regression(config, y_trn, y_trn_pred, loggers, 'trn', is_log=True, is_save=False)
        for m in metrics_trn.index.values:
            cv_progress.at[fold_idx, f"trn_{m}"] = metrics_trn.at[m, 'trn']
        metrics_val = eval_regression(config, y_val, y_val_pred, loggers, 'val', is_log=True, is_save=False)
        for m in metrics_val.index.values:
            cv_progress.at[fold_idx, f"val_{m}"] = metrics_val.at[m, 'val']
        if is_tst:
            metrics_tst = eval_regression(config, y_tst, y_tst_pred, loggers, 'tst', is_log=True, is_save=False)
            for m in metrics_tst.index.values:
                cv_progress.at[fold_idx, f"tst_{m}"] = metrics_tst.at[m, 'tst']

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

        if config.optimized_part == "trn":
            metrics_main = metrics_trn
        elif config.optimized_part == "val":
            metrics_main = metrics_val
        elif config.optimized_part == "tst":
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
            if Path(f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt").is_file():
                if config.model_type == "widedeep_tab_mlp":
                    model = WDTabMLPModel.load_from_checkpoint(checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                    model.eval()
                    model.freeze()
                elif config.model_type == "widedeep_tab_resnet":
                    model = WDTabResnetModel.load_from_checkpoint(checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                    model.eval()
                    model.freeze()
                elif config.model_type == "pytorch_tabular_autoint":
                    model = PTAutoIntModel.load_from_checkpoint(checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                    model.eval()
                    model.freeze()
                elif config.model_type == "pytorch_tabular_tabnet":
                    model = PTTabNetModel.load_from_checkpoint(checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                    model.eval()
                    model.freeze()
                elif config.model_type == "pytorch_tabular_node":
                    model = PTNODEModel.load_from_checkpoint(checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                    model.eval()
                    model.freeze()
                elif config.model_type == "pytorch_tabular_category_embedding":
                    model = PTCategoryEmbeddingModel.load_from_checkpoint(checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                    model.eval()
                    model.freeze()
                else:
                    raise ValueError(f"Unsupported model: {config.model_type}")
            best["model"] = model
            best["trainer"] = trainer

            def predict_func(X):
                X = np.float32(X)
                X = torch.from_numpy(X)
                tmp = best["model"](X)
                return tmp.cpu().detach().numpy()

            best['predict_func'] = predict_func
            best['feature_importances'] = feature_importances
            best['fold'] = fold_idx
            best['ids_trn'] = ids_trn
            best['ids_val'] = ids_val
            df.loc[df.index[ids_trn], "Estimation"] = y_trn_pred
            df.loc[df.index[ids_val], "Estimation"] = y_val_pred
            if is_tst:
                df.loc[df.index[ids_tst], "Estimation"] = y_tst_pred

        cv_progress.at[fold_idx, 'fold'] = fold_idx
        cv_progress.at[fold_idx, 'optimized_metric'] = metrics_main.at[config.optimized_metric, config.optimized_part]

    df = df.astype({"Estimation": 'float32'})
    cv_progress.to_excel(f"cv_progress.xlsx", index=False)
    cv_ids = df.loc[:, [f"fold_{fold_idx:04d}" for fold_idx in cv_progress.loc[:, 'fold'].values]]
    cv_ids.to_excel(f"cv_ids.xlsx", index=True)
    predictions = df.loc[:, [f"fold_{best['fold']:04d}", target_name, "Estimation"]]
    predictions.to_excel(f"predictions.xlsx", index=True)

    datamodule.ids_trn = best['ids_trn']
    datamodule.ids_val = best['ids_val']

    datamodule.plot_split(f"_best_{best['fold']:04d}")

    y_trn = df.loc[df.index[datamodule.ids_trn], target_name].values
    y_trn_pred = df.loc[df.index[datamodule.ids_trn], "Estimation"].values
    y_val = df.loc[df.index[datamodule.ids_val], target_name].values
    y_val_pred = df.loc[df.index[datamodule.ids_val], "Estimation"].values
    if is_tst:
        y_tst = df.loc[df.index[datamodule.ids_tst], target_name].values
        y_tst_pred = df.loc[df.index[datamodule.ids_tst], "Estimation"].values

    metrics_trn = eval_regression(config, y_trn, y_trn_pred, None, 'trn', is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
    metrics_names = metrics_trn.index.values
    metrics_trn_cv = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names] + [f"{x}_cv_std" for x in metrics_names], columns=['trn'])
    for metric in metrics_names:
        metrics_trn_cv.at[f"{metric}_cv_mean", 'trn'] = cv_progress[f"trn_{metric}"].mean()
        metrics_trn_cv.at[f"{metric}_cv_std", 'trn'] = cv_progress[f"trn_{metric}"].std()
    metrics_trn = pd.concat([metrics_trn, metrics_trn_cv])
    metrics_trn.to_excel(f"metrics_trn_best_{best['fold']:04d}.xlsx", index=True, index_label="metric")

    metrics_val = eval_regression(config, y_val, y_val_pred, None, 'val', is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
    metrics_val_cv = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names] + [f"{x}_cv_std" for x in metrics_names], columns=['val'])
    for metric in metrics_names:
        metrics_val_cv.at[f"{metric}_cv_mean", 'val'] = cv_progress[f"val_{metric}"].mean()
        metrics_val_cv.at[f"{metric}_cv_std", 'val'] = cv_progress[f"val_{metric}"].std()
    metrics_val = pd.concat([metrics_val, metrics_val_cv])
    metrics_val.to_excel(f"metrics_val_best_{best['fold']:04d}.xlsx", index=True, index_label="metric")

    if is_tst:
        metrics_tst = eval_regression(config, y_tst, y_tst_pred, None, 'tst', is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
        metrics_tst_cv = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names] + [f"{x}_cv_std" for x in metrics_names], columns=['tst'])
        for metric in metrics_names:
            metrics_tst_cv.at[f"{metric}_cv_mean", 'tst'] = cv_progress[f"tst_{metric}"].mean()
            metrics_tst_cv.at[f"{metric}_cv_std", 'tst'] = cv_progress[f"tst_{metric}"].std()
        metrics_tst = pd.concat([metrics_tst, metrics_tst_cv])

        metrics_val_tst_cv_mean = pd.DataFrame(index=[f"{x}_cv_mean_val_tst" for x in metrics_names],columns=['val', 'tst'])
        for metric in metrics_names:
            val_tst_value = 0.5 * (metrics_val.at[f"{metric}_cv_mean", 'val'] + metrics_tst.at[f"{metric}_cv_mean", 'tst'])
            metrics_val_tst_cv_mean.at[f"{metric}_cv_mean_val_tst", 'val'] = val_tst_value
            metrics_val_tst_cv_mean.at[f"{metric}_cv_mean_val_tst", 'tst'] = val_tst_value
        metrics_val = pd.concat([metrics_val, metrics_val_tst_cv_mean.loc[:, ['val']]])
        metrics_tst = pd.concat([metrics_tst, metrics_val_tst_cv_mean.loc[:, ['tst']]])
        metrics_val.to_excel(f"metrics_val_best_{best['fold']:04d}.xlsx", index=True, index_label="metric")
        metrics_tst.to_excel(f"metrics_tst_best_{best['fold']:04d}.xlsx", index=True, index_label="metric")

    if config.optimized_part == "trn":
        metrics_main = metrics_trn
    elif config.optimized_part == "val":
        metrics_main = metrics_val
    elif config.optimized_part == "tst":
        metrics_main = metrics_tst
    else:
        raise ValueError(f"Unsupported config.optimized_part: {config.optimized_part}")

    if best['feature_importances'] is not None:
        save_feature_importance(best['feature_importances'], config.num_top_features)

    formula = f"Estimation ~ {target_name}"
    model_linear = smf.ols(formula=formula, data=df.loc[df.index[datamodule.ids_trn], :]).fit()
    df.loc[df.index[datamodule.ids_trn], "Estimation acceleration"] = df.loc[df.index[datamodule.ids_trn], "Estimation"].values - model_linear.predict(df.loc[df.index[datamodule.ids_trn], :])
    df.loc[df.index[datamodule.ids_val], "Estimation acceleration"] = df.loc[df.index[datamodule.ids_val], "Estimation"].values - model_linear.predict(df.loc[df.index[datamodule.ids_val], :])
    if is_tst:
        df.loc[df.index[datamodule.ids_tst], "Estimation acceleration"] = df.loc[df.index[datamodule.ids_tst], "Estimation"].values - model_linear.predict(df.loc[df.index[datamodule.ids_tst], :])
    fig = go.Figure()
    add_scatter_trace(fig, df.loc[df.index[datamodule.ids_trn], target_name].values, df.loc[df.index[datamodule.ids_trn], "Estimation"].values, f"Train")
    add_scatter_trace(fig, df.loc[df.index[datamodule.ids_trn], target_name].values, model_linear.fittedvalues.values, "", "lines")
    add_scatter_trace(fig, df.loc[df.index[datamodule.ids_val], target_name].values, df.loc[df.index[datamodule.ids_val], "Estimation"].values, f"Val")
    if is_tst:
        add_scatter_trace(fig, df.loc[df.index[datamodule.ids_tst], target_name].values, df.loc[df.index[datamodule.ids_tst], "Estimation"].values, f"Test")
    add_layout(fig, target_name, f"Estimation", f"")
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
    if is_tst:
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
    if is_tst:
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

    expl_data = {
        'model': best["model"],
        'predict_func': best['predict_func'],
        'df': df,
        'feature_names': feature_names,
        'outcome_name': target_name,
        'ids_all': np.arange(df.shape[0]),
        'ids_trn': datamodule.ids_trn,
        'ids_val': datamodule.ids_val,
        'ids_tst': datamodule.ids_tst
    }
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
