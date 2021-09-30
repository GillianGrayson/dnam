import numpy as np
from src.models.dnam.tabnet import TabNetModel
import torch
import os
import matplotlib.pyplot as plt
import pickle
from src.datamodules.datasets.dnam_dataset import DNAmDataset
from torch.utils.data import DataLoader
from scipy.sparse import csc_matrix
from scipy.special import log_softmax
from scipy.special import softmax
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.violin import add_violin_trace
from scripts.python.routines.plot.layout import add_layout
import plotly.graph_objects as go
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
from glob import glob
import plotly.express as px


log = utils.get_logger(__name__)

dotenv.load_dotenv(override=True)

@hydra.main(config_path="../configs/", config_name="main_xai.yaml")
def main(config: DictConfig):

    # class_names = [
    #     "Control",
    #     "Schizophrenia",
    #     "Depression",
    #     "Parkinson"
    # ]

    # class_names = [
    #     "Schizophrenia Control",
    #     "Schizophrenia",
    #     "Depression Control",
    #     "Depression",
    #     "Parkinson Control",
    #     "Parkinson"
    # ]

    class_names = [
        "Schizophrenia",
        "Depression",
        "Parkinson"
    ]

    model = "tabnetpl_unnhpc"
    num_feat = 391023
    folder_path = f"E:/YandexDisk/Work/pydnameth/datasets/meta/SchizophreniaDepressionParkinsonCases/{num_feat}/models/{model}/logs/multiruns/2021-09-30_03-57-27"

    num_top_features = config.num_top_features
    runs = next(os.walk(folder_path))[1]
    runs.sort()

    feat_importances = pd.DataFrame(data=np.zeros(shape=(num_feat, len(runs)), dtype=float), columns=runs)
    for run_id, run in enumerate(runs):
        checkpoint_fn = glob(f"{folder_path}/{run}/checkpoints/*.ckpt")

        if "seed" in config:
            seed_everything(config.seed)

        model = TabNetModel.load_from_checkpoint(checkpoint_path=checkpoint_fn[0])
        model.produce_probabilities = True
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

        if run_id == 0:
            feat_importances.loc[:, 'feat'] = datamodule.betas.columns.values

        background_dataloader = train_dataloader

        fis = np.zeros((model.hparams.input_dim))
        for data, targets, indexes in tqdm(background_dataloader):
            M_explain, masks = model.forward_masks(data)
            fis += M_explain.sum(dim=0).cpu().detach().numpy()

        fis = fis / np.sum(fis)
        feat_importances.loc[:, run] = fis

        if run_id == 0:
            common_df = pd.merge(datamodule.pheno, datamodule.betas, left_index=True, right_index=True)
            curr_var = np.var(common_df.loc[:, 'cg00000109'].values)
            olol = 1

    feat_importances.set_index('feat', inplace=True)
    feat_importances['average'] = feat_importances.mean(numeric_only=True, axis=1)
    feat_importances.sort_values(['average'], ascending=[False], inplace=True)
    feat_importances.to_excel("./feat_importances.xlsx", index=True)

    for feat_id, feat in enumerate(feat_importances.index.values[0:num_top_features]):

        curr_var = np.var(common_df.loc[:, feat].values)

        fig = go.Figure()
        for cl_id, cl in enumerate(class_names):
            add_violin_trace(fig, common_df.loc[common_df['Status'] == cl_id, feat].values, cl)
        add_layout(fig, "", f"{feat}", f"variance = {curr_var:0.2e}")
        fig.update_layout({'colorway': px.colors.qualitative.Set1})
        Path(f"features/violin").mkdir(parents=True, exist_ok=True)
        save_figure(fig, f"features/violin/{feat_id}_{feat}")


if __name__ == "__main__":
    main()
