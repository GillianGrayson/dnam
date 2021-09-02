import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    Trainer,
    seed_everything,
)
import pandas as pd
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score


def train(config: DictConfig):

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()

    # Init lightning trainer
    trainer: Trainer = hydra.utils.instantiate(config.trainer, _convert_="partial")

    num_col_names = list(datamodule.betas.columns.values)
    cat_col_names = []
    data = pd.merge(datamodule.pheno.loc[:, datamodule.outcome], datamodule.betas, left_index=True, right_index=True)

    train_data = data.iloc[datamodule.ids_train]
    val_data = data.iloc[datamodule.ids_val]
    test_data = data.iloc[datamodule.ids_test]

    lr: 0.0001
    weight_decay: 0.00005

    model = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                             optimizer_params=dict(lr=0.001, weight_decay=0.0005),
                             verbose=1,
                             scheduler_params={"step_size": 10, "gamma": 0.9},
                             scheduler_fn=torch.optim.lr_scheduler.StepLR,
                             mask_type="sparsemax"#'entmax'  #
                             )

    X_train = train_data.loc[:, datamodule.betas.columns.values].values
    y_train = train_data.loc[:, datamodule.outcome].values

    X_val = val_data.loc[:, datamodule.betas.columns.values].values
    y_val = val_data.loc[:, datamodule.outcome].values

    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_name=['train', 'valid'],
        eval_metric=['accuracy'],
        max_epochs=1000, patience=100,
        batch_size=128, virtual_batch_size=64,
        num_workers=0,
        weights=1,
        drop_last=False
    )

    ololo = 1


