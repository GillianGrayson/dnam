import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_tabnet.tab_model import TabNetClassifier
from models_sa.metrics_multiclass import get_metrics_dict
from models_sa.tabnet.logging import log_hyperparameters
from models_sa.tabnet.callbacks import get_custom_callback
from src.utils import utils
from typing import List
import lightgbm as lgb


log = utils.get_logger(__name__)

def train_lightgbm(config: DictConfig):

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    config.logger.wandb["project"] = config.project_name

    if config.model.objective == "multiclass":
        metric = {'multi_logloss'}
    else:
        raise ValueError(f"Unsupported model objective: {config.model.objective}")

    model_params = {
        'num_class': config.model.output_dim,
        'objective': config.model.objective,
        'boosting': config.model.boosting,
        'num_iterations': config.model.num_iterations,
        'learning_rate': config.model.learning_rate,
        'num_leaves': config.model.num_leaves,
        'device': config.model.device,
        'max_depth': config.model.max_depth,
        'min_data_in_leaf': config.model.min_data_in_leaf,
        'metric': metric,
        'feature_fraction': config.model.feature_fraction,
        'bagging_fraction': config.model.bagging_fraction,
        'bagging_freq': config.model.bagging_freq,
        'verbose': config.model.verbose
    }

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()
    feature_names = datamodule.betas.columns.to_list()
    data = pd.merge(datamodule.pheno.loc[:, datamodule.outcome], datamodule.betas, left_index=True, right_index=True)
    train_data = data.iloc[datamodule.ids_train]
    val_data = data.iloc[datamodule.ids_val]
    test_data = data.iloc[datamodule.ids_test]
    X_train = train_data.loc[:, datamodule.betas.columns.values].values
    y_train = train_data.loc[:, datamodule.outcome].values
    X_val = val_data.loc[:, datamodule.betas.columns.values].values
    y_val = val_data.loc[:, datamodule.outcome].values
    X_test = test_data.loc[:, datamodule.betas.columns.values].values
    y_test = test_data.loc[:, datamodule.outcome].values
    ds_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train, feature_name=feature_names)
    ds_test = lgb.Dataset(X_test, label=y_test, reference=ds_train, feature_name=feature_names)

    max_epochs = config.trainer.max_epochs
    patience = config.trainer.patience

    evals_result = {}  # to record eval results for plotting

    bst = lgb.train(
        params=model_params,
        train_set=ds_train,
        num_boost_round=max_epochs,
        valid_sets=[ds_val, ds_train],
        valid_names=['val', 'train'],
        evals_result=evals_result,
        early_stopping_rounds=patience,
        verbose_eval=True
    )

    bst.save_model(f"epoch_{bst.best_iteration}.txt", num_iteration=bst.best_iteration)

    print('Plotting metrics recorded during training...')
    ax = lgb.plot_metric(evals_result)
    plt.show()

    print('Plotting feature importances...')
    ax = lgb.plot_importance(bst, max_num_features=10)
    plt.show()

    feature_importances = pd.DataFrame.from_dict({'feature': bst.feature_name(), 'importance': list(bst.feature_importance())})
    feature_importances.set_index('feature', inplace=True)
    feature_importances.to_excel("./feature_importances.xlsx", index=True)

    metrics_dict = get_metrics_dict(config.model.output_dim, object)
    metrics = [
        metrics_dict["accuracy_macro"],
        metrics_dict["accuracy_weighted"],
        metrics_dict["f1_macro"],
        metrics_dict["cohen_kappa"],
        metrics_dict["matthews_corrcoef"],
        metrics_dict["f1_weighted"],
    ]

    y_train_pred_probs = bst.predict(X_train, num_iteration=bst.best_iteration)
    y_val_pred_probs = bst.predict(X_val, num_iteration=bst.best_iteration)
    y_test_pred_probs = bst.predict(X_test, num_iteration=bst.best_iteration)

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return 0

