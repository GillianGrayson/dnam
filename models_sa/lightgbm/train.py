import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)
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

def train_tabnet(config: DictConfig):

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    config.logger.wandb["project"] = config.project_name

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()
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
    ds_train = lgb.Dataset(X_train, label=y_train, feature_name=datamodule.betas.columns.values)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train, feature_name=datamodule.betas.columns.values)
    ds_test = lgb.Dataset(X_test, label=y_test, reference=ds_train, feature_name=datamodule.betas.columns.values)

    metrics_dict = get_metrics_dict(config.model.n_output, object)
    metrics = [
        metrics_dict["accuracy_macro"],
        metrics_dict["accuracy_weighted"],
        metrics_dict["f1_macro"],
        metrics_dict["cohen_kappa"],
        metrics_dict["matthews_corrcoef"],
        metrics_dict["f1_weighted"],
    ]

    model_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': {'multi_logloss'},
        'num_class': config.model.output_dim,
        'max_depth': 7,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    max_epochs = config.trainer.max_epochs
    patience = config.trainer.patience

    bst = lgb.train(
        params=model_params,
        train_set=ds_train,
        num_boost_round=max_epochs,
        valid_sets=[ds_val],
        valid_names=['val'],
        early_stopping_rounds=patience,
        verbose_eval=True
    )

    bst.save_model(f"epoch_{bst.best_iteration}.txt", num_iteration=bst.best_iteration)

    y_test_pred_probs = bst.predict(ds_test, num_iteration=bst.best_iteration)

    # Init model
    model = TabNetClassifier(
        n_d=config.model.n_d,
        n_a=config.model.n_a,
        n_steps=config.model.n_steps,
        n_independent=config.model.n_independent,
        n_shared=config.model.n_shared,
        momentum=config.model.momentum,
        lambda_sparse=config.model.lambda_sparse,
        seed=config.model.seed,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=config.model.optimizer_lr, weight_decay=config.model.optimizer_weight_decay),
        verbose=1,
        scheduler_params={"step_size": config.model.scheduler_step_size, "gamma": config.model.scheduler_gamma},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type=config.model.mask_type,
        device_name=config.model.device_name
    )






    TabNetCallback = get_custom_callback()

    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_test, y_test), (X_val, y_val)],
        eval_name=['train', 'test', 'val'],
        eval_metric=eval_metric,
        max_epochs=config.trainer.max_epochs,
        patience=config.trainer.patience,
        batch_size=config.trainer.batch_size,
        virtual_batch_size=config.trainer.virtual_batch_size,
        callbacks=[TabNetCallback],
        num_workers=0,
        weights=1,
        drop_last=False
    )

    # save tabnet model
    saved_filepath = model.save_model("./best_model")
    print(saved_filepath)

    metrics_dict = model.history.history
    metrics_dict['epoch'] = list(range(1, len(metrics_dict['loss']) + 1))
    metrics = pd.DataFrame.from_dict(model.history.history)
    metrics.set_index('epoch', inplace=True)
    metrics.to_excel("./metrics.xlsx", index=True)

    feature_importances = pd.DataFrame.from_dict({'feature': datamodule.betas.columns.values, 'importance': model.feature_importances_})
    feature_importances.set_index('feature', inplace=True)
    feature_importances.to_excel("./feature_importances.xlsx", index=True)

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return max(model.history.history[optimized_metric])

