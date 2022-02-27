import numpy as np
from src.models.dnam.tabnet import TabNetModel
import torch
import os
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from src.datamodules.datasets.dnam_dataset import DNAmDataset
from torch.utils.data import DataLoader
from scipy.sparse import csc_matrix
from scipy.special import log_softmax
from scipy.special import softmax
import plotly.graph_objects as go
from collections import Counter
import lightgbm as lgb
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout
from tqdm import tqdm
import pandas as pd
import dotenv
from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pytorch_lightning import seed_everything
import hydra
from src.utils import utils
from omegaconf import DictConfig
import shap
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
import plotly.figure_factory as ff
from scripts.python.routines.manifest import get_manifest
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)
from sa.logging import log_hyperparameters
from pytorch_lightning.loggers import LightningLoggerBase
from src.utils import utils
from scripts.python.routines.plot.save import save_figure
from sa.classification.routines import eval_classification, eval_loss
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.layout import add_layout
from typing import List
import wandb
from catboost import CatBoost
import xgboost as xgb


log = utils.get_logger(__name__)

def inference(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

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

    if config.sa_model == "lightgbm":
        model = lgb.Booster(model_file=config.ckpt_path)
    if config.sa_model == "catboost":
        model = CatBoost()
        model.load_model(config.ckpt_path)
    if config.sa_model == "xgboost":
        model = xgb.Booster()
        model.load_model(config.ckpt_path)
    elif config.sa_model == "tabnetpl":
        model = TabNetModel.load_from_checkpoint(checkpoint_path=f"{config.ckpt_path}")
        model.produce_probabilities = True
        model.eval()
        model.freeze()
    else:
        raise ValueError(f"Unsupported sa_model")

    test_datasets = {
        'GSE87571': ['Control'],
    }

    for dataset in test_datasets:

        # Init Lightning datamodule for test
        log.info(f"Instantiating datamodule <{config.datamodule._target_}> for {dataset}")
        config.datamodule.dnam_fn = f"mvals_{dataset}.pkl"
        config.datamodule.pheno_fn = f"pheno_{dataset}.pkl"
        datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
        datamodule.setup()
        feature_names = datamodule.dnam.columns.to_list()
        dataloader_test = datamodule.test_dataloader()
        data = pd.merge(datamodule.pheno.loc[:, datamodule.outcome], datamodule.dnam, left_index=True, right_index=True)
        test_data = data.iloc[datamodule.ids]
        X_test = test_data.loc[:, datamodule.dnam.columns.values].values
        y_test = test_data.loc[:, datamodule.outcome].values

        if config.sa_model == "lightgbm":
            y_test_pred_probs = model.predict(X_test)
        if config.sa_model == "catboost":
            y_test_pred_probs = model.predict(X_test)
        if config.sa_model == "xgboost":
            dmat_test = xgb.DMatrix(X_test, y_test, feature_names=datamodule.dnam.columns.values)
            y_test_pred_probs = model.predict(dmat_test)
        elif config.sa_model == "tabnetpl":
            X_test_pt = torch.from_numpy(X_test)
            y_test_pred_probs = model(X_test_pt).cpu().detach().numpy()
        else:
            raise ValueError(f"Unsupported sa_model")

        y_test_pred = np.argmax(y_test_pred_probs, 1)

        class_names = list(datamodule.statuses.keys())

        eval_classification(config, dataset, class_names, y_test, y_test_pred, y_test_pred_probs, loggers, probs=False)

    for logger in loggers:
        logger.save()
    wandb.finish()
