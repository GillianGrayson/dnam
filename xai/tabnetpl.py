import numpy as np
from src.models.dnam.tabnet import TabNetModel
import torch
import os
import matplotlib.pyplot as plt
import pickle
from src.datamodules.datasets.dnam_dataset import DNAmDataset
from torch.utils.data import DataLoader
from scipy.sparse import csc_matrix
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


log = utils.get_logger(__name__)

dotenv.load_dotenv(override=True)

@hydra.main(config_path="../configs/", config_name="main_xai.yaml")
def main(config: DictConfig):

    num_top_features = 20

    if "seed" in config:
        seed_everything(config.seed)

    checkpoint_path = config.checkpoint_path
    checkpoint_name = config.checkpoint_name

    model = TabNetModel.load_from_checkpoint(checkpoint_path=f"{checkpoint_path}/{checkpoint_name}")
    model.eval()
    model.freeze()

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    dataset = ConcatDataset([train_dataloader.dataset, val_dataloader.dataset, test_dataloader.dataset])
    all_dataloader = DataLoader(
        dataset,
        batch_size=config.datamodule.batch_size,
        num_workers=config.datamodule.num_workers,
        pin_memory=config.datamodule.pin_memory,
        shuffle=True
    )

    feature_importances = np.zeros((model.hparams.input_dim))
    for data, targets, indexes in tqdm(train_dataloader):
        # data = data.to(model.device).float()
        M_explain, masks = model.forward_masks(data)
        feature_importances += M_explain.sum(dim=0).cpu().detach().numpy()

    #feature_importances = csc_matrix.dot(feature_importances, model.reducing_matrix)
    feature_importances = feature_importances / np.sum(feature_importances)
    feature_importances_df = pd.DataFrame.from_dict(
        {
            'feature': datamodule.betas.columns.values,
            'importance': feature_importances
        }
    )
    feature_importances_df.set_index('feature', inplace=True)
    feature_importances_df.to_excel("./feature_importances.xlsx", index=True)


    batch_id = 0
    d = {}
    for background, outs_real, indexes in tqdm(train_dataloader):

        outs_pred = model(background).flatten()

        if batch_id == 0:
            e = shap.DeepExplainer(model, background)

        shap_values = e.shap_values(background)

        if batch_id == 0:
            shap_abs = np.absolute(shap_values)
            shap_mean_abs = np.mean(shap_abs, axis=0)
            order = np.argsort(shap_mean_abs)[::-1]
            features = datamodule.betas.columns.values
            features_best = features[order[0:num_top_features]]

        subject_indices = indexes.flatten().cpu().detach().numpy()
        subjects = datamodule.betas.index.values[subject_indices]
        outcomes = datamodule.pheno.loc[subjects, config.datamodule.outcome].to_numpy()

        betas = background.cpu().detach().numpy()
        preds = outs_pred.cpu().detach().numpy()

        if batch_id == 0:
            d['subject'] = subjects
            d['outcome'] = outcomes
            d['preds'] = preds

            for f_id in range(0, num_top_features):
                feat = features_best[f_id]
                curr_beta = betas[:, order[f_id]]
                curr_shap = shap_values[:, order[f_id]]
                d[f"{feat}_beta"] = curr_beta
                d[f"{feat}_shap"] = curr_shap
        else:
            d['subject'] = np.append(d['subject'], subjects)
            d['outcome'] = np.append(d['outcome'], outcomes)
            d['preds'] = np.append(d['preds'], preds)

            for f_id in range(0, num_top_features):
                feat = features_best[f_id]
                curr_beta = betas[:, order[f_id]]
                curr_shap = shap_values[:, order[f_id]]
                d[f"{feat}_beta"] = np.append(d[f"{feat}_beta"], curr_beta)
                d[f"{feat}_shap"] = np.append(d[f"{feat}_shap"], curr_shap)

        batch_id += 1

    df_features = pd.DataFrame(d)
    df_features.to_excel(f"shap_values_{config.datamodule.batch_size}_{num_top_features}.xlsx", index=False)


if __name__ == "__main__":
    main()
