import numpy as np
from src.models.dnam.fcmlp import FCMLPModel
import os
import matplotlib.pyplot as plt
import pickle
from src.datamodules.datasets.dnam_dataset import DNAmDataset
from torch.utils.data import DataLoader
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
from pytorch_tabnet.tab_model import TabNetClassifier


log = utils.get_logger(__name__)

dotenv.load_dotenv(override=True)

@hydra.main(config_path="../configs/", config_name="main_xai.yaml")
def main(config: DictConfig):

    num_top_features = 20

    if "seed" in config:
        seed_everything(config.seed)

    checkpoint_path = config.checkpoint_path
    checkpoint_name = config.checkpoint_name

    model = TabNetClassifier()
    model.load_model(f"{checkpoint_path}/{checkpoint_name}")

    # Init Lightning datamodule
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

    background = X_val

    explainer = shap.Explainer(model.predict, background)
    shap_values = explainer(background)



    ololo = 1


if __name__ == "__main__":
    main()
