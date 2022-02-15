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


log = utils.get_logger(__name__)

dotenv.load_dotenv(override=True)

@hydra.main(config_path="../../configs/", config_name="main.yaml")
def main(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

    model = TabNetModel.load_from_checkpoint(checkpoint_path=f"{config.ckpt_path}/{config.ckpt_name}")
    model.produce_probabilities = True
    model.eval()
    model.freeze()

    # Init Lightning datamodule for train and val
    log.info(f"Instantiating datamodule <{config.datamodule_train_val._target_}>")
    datamodule_val: LightningDataModule = hydra.utils.instantiate(config.datamodule_train_val)
    datamodule_val.setup()
    dataloader_val = datamodule_val.val_dataloader()
    df_val = pd.merge(datamodule_val.pheno, datamodule_val.betas, left_index=True, right_index=True)
    statuses = datamodule_val.statuses

    test_datasets = {
        'GSE116379': ['Control', 'Schizophrenia'],
        'GSE113725': ['Control', 'Depression'],
        'GSE41169': ['Control', 'Schizophrenia'],
        'GSE116378': ['Control', 'Schizophrenia'],
    }

    for dataset in test_datasets:

        # Init Lightning datamodule for test
        log.info(f"Instantiating datamodule <{config.datamodule_test._target_}> for {dataset}")
        config.datamodule_test.dnam_fn = f"mvals_{dataset}_regRCPqn.pkl"
        config.datamodule_test.pheno_fn = f"pheno_{dataset}.pkl"
        datamodule_test: LightningDataModule = hydra.utils.instantiate(config.datamodule_test)
        datamodule_test.setup()
        dataloader_test = datamodule_test.test_dataloader()
        df_test = pd.merge(datamodule_test.pheno, datamodule_test.betas, left_index=True, right_index=True)

        outs_real_all = np.empty(0, dtype=int)
        outs_pred_all = np.empty(0, dtype=int)
        outs_prob_all = np.empty(shape=(0, len(statuses)), dtype=int)
        for x, outs_real, indexes in tqdm(dataloader_test):
            outs_real = outs_real.cpu().detach().numpy()
            outs_prob = model(x).cpu().detach().numpy()
            outs_pred = np.argmax(outs_prob, axis=1)
            outs_real_all = np.append(outs_real_all, outs_real, axis=0)
            outs_pred_all = np.append(outs_pred_all, outs_pred, axis=0)
            outs_prob_all = np.append(outs_prob_all, outs_prob, axis=0)

        conf_mtx = confusion_matrix(outs_real_all, outs_pred_all, labels=list(range(len(statuses))))
        fig = ff.create_annotated_heatmap(conf_mtx, x=list(statuses.keys()), y=list(statuses.keys()), colorscale='Viridis')
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=0.5,
                                y=-0.1,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=-0.33,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        fig.update_layout(margin=dict(t=50, l=200))
        fig['data'][0]['showscale'] = True
        save_figure(fig, f'confusion_matrix_{dataset}')


if __name__ == "__main__":
    main()
