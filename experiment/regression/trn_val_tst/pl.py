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
from pytorch_lightning.loggers import LightningLoggerBase
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.layout import add_layout
import numpy as np
from src.utils import utils
import pandas as pd
from experiment.routines import plot_confusion_matrix
from experiment.multiclass.shap import perform_shap_explanation


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
        if len(test_dataloader) > 0:
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
    class_names = datamodule.get_class_names()
    raw_data = datamodule.get_raw_data()
    X_train = torch.from_numpy(raw_data['X_train'])
    y_train = raw_data['y_train']
    X_val = torch.from_numpy(raw_data['X_val'])
    y_val = raw_data['y_val']
    X_test = torch.from_numpy(raw_data['X_test'])
    y_test = raw_data['y_test']

    model.eval()
    model.freeze()

    def shap_proba(X):
        model.produce_probabilities = True
        X = torch.from_numpy(X)
        tmp = model(X)
        return tmp.cpu().detach().numpy()

    model.produce_probabilities = True
    y_train_pred_probs = model(X_train).cpu().detach().numpy()
    y_val_pred_probs = model(X_val).cpu().detach().numpy()
    y_test_pred_probs = model(X_test).cpu().detach().numpy()
    model.produce_probabilities = False
    y_train_pred_raw = model(X_train).cpu().detach().numpy()
    y_val_pred_raw = model(X_val).cpu().detach().numpy()
    y_test_pred_raw = model(X_test).cpu().detach().numpy()
    y_train_pred = np.argmax(y_train_pred_probs, 1)
    y_val_pred = np.argmax(y_val_pred_probs, 1)
    y_test_pred = np.argmax(y_test_pred_probs, 1)

    raw_data['y_train_pred_probs'] = y_train_pred_probs
    raw_data['y_val_pred_probs'] = y_val_pred_probs
    raw_data['y_test_pred_probs'] = y_test_pred_probs
    raw_data['y_train_pred_raw'] = y_train_pred_raw
    raw_data['y_val_pred_raw'] = y_val_pred_raw
    raw_data['y_test_pred_raw'] = y_test_pred_raw
    raw_data['y_train_pred'] = y_train_pred
    raw_data['y_val_pred'] = y_val_pred
    raw_data['y_test_pred'] = y_test_pred

    if config.model._target_ == "src.models.dnam.tabnet.TabNetModel":
        feature_importances = np.zeros((model.hparams.input_dim))
        M_explain, masks = model.forward_masks(X_train)
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

    plot_confusion_matrix(y_train, y_train_pred, class_names, "train")
    plot_confusion_matrix(y_val, y_val_pred, class_names, "val")
    plot_confusion_matrix(y_test, y_test_pred, class_names, "test")

    if config.is_shap == True:
        X_all = np.concatenate((X_train, X_val, X_test))
        y_all = np.concatenate((y_train, y_val, y_test))
        y_all_pred = np.concatenate((y_train_pred, y_val_pred, y_test_pred))
        y_all_pred_raw = np.concatenate((y_train_pred_raw, y_val_pred_raw, y_test_pred_raw))
        y_all_pred_probs = np.concatenate((y_train_pred_probs, y_val_pred_probs, y_test_pred_probs))
        ids_train = np.linspace(0, X_train.shape[0], X_train.shape[0], dtype=int)
        ids_val = np.linspace(X_train.shape[0], X_train.shape[0] + X_val.shape[0], X_val.shape[0], dtype=int)
        ids_test = np.linspace(X_train.shape[0] + X_val.shape[0], X_train.shape[0] + X_val.shape[0] + X_test.shape[0], X_test.shape[0], dtype=int)
        raw_data['X_all'] = X_all
        raw_data['y_all'] = y_all
        raw_data['y_all_pred'] = y_all_pred
        raw_data['y_all_pred_probs'] = y_all_pred_probs
        raw_data['y_all_pred_raw'] = y_all_pred_raw
        raw_data['ids_train'] = ids_train
        raw_data['ids_val'] = ids_val
        raw_data['ids_test'] = ids_test
        perform_shap_explanation(config, model, shap_proba, raw_data, feature_names, class_names)

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
