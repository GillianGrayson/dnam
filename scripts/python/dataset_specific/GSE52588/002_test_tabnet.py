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
import pathlib
from src.datamodules.dnam.betas_pheno import BetasPhenoDataModule


dataset = "GSE52588"

path = f"E:/YandexDisk/Work/pydnameth/datasets"
path_background = "E:/YandexDisk/Work/pydnameth/datasets/meta/121da597d6d3fe7b3b1b22a0ddc26e61"
datasets_info = pd.read_excel(f"{path}/datasets.xlsx", index_col='dataset')
platform = datasets_info.loc[dataset, 'platform']
path_save = f"{path}/{platform}/{dataset}/special/002_test_tabnet"
os.chdir(path_save)
pathlib.Path(f"{path_save}/figs").mkdir(parents=True, exist_ok=True)

checkpoint_path = f"{path_background}/models/24829/tabnetpl_unnhpc_average_all_340/logs/multiruns/2021-11-19_20-26-52/2/checkpoints"
checkpoint_name = "414.ckpt"
model = TabNetModel.load_from_checkpoint(checkpoint_path=f"{checkpoint_path}/{checkpoint_name}")
model.produce_probabilities = True
model.eval()
model.freeze()

datamodule = BetasPhenoDataModule(
    path=f"{path}/{platform}/{dataset}/special/001_generate_dataset",
    cpgs_fn=f"{path_background}/cpgs/24829/tabnetpl/average/all/340.xlsx",
    statuses_fn=f"{path}/{platform}/{dataset}/special/001_generate_dataset/statuses.xlsx",
    outcome="Status",
    batch_size=500,
    train_val_test_split=(0.7, 0.15, 0.15),
    num_workers=0,
    pin_memory=False,
    seed=2,
    weighted_sampler=False,
)
datamodule.setup()

train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()
test_dataloader = datamodule.test_dataloader()
dataset = ConcatDataset([train_dataloader.dataset, val_dataloader.dataset, test_dataloader.dataset])
all_dataloader = DataLoader(
    dataset,
    batch_size=datamodule.batch_size,
    num_workers=datamodule.num_workers,
    pin_memory=datamodule.pin_memory,
    shuffle=True
)
common_df = pd.merge(datamodule.pheno, datamodule.betas, left_index=True, right_index=True)

outs_real_all = np.empty(0, dtype=int)
outs_pred_all = np.empty(0, dtype=int)
outs_prob_all = np.empty(shape=(0, 4), dtype=int)
indexes_all = np.empty(0, dtype=int)
for x, outs_real, indexes in tqdm(all_dataloader):
    outs_real = outs_real.cpu().detach().numpy()
    outs_prob = model(x).cpu().detach().numpy()
    outs_pred = np.argmax(outs_prob, axis=1)
    indexes_all =np.append(indexes_all, indexes, axis=0)
    outs_real_all = np.append(outs_real_all, outs_real, axis=0)
    outs_pred_all = np.append(outs_pred_all, outs_pred, axis=0)
    outs_prob_all = np.append(outs_prob_all, outs_prob, axis=0)

res_dict = {'Index': indexes_all, 'Real': outs_real_all, 'Pred': outs_pred_all}
res_df = pd.DataFrame(res_dict)
res_df.set_index('Index', inplace=True)
res_df.to_excel(f"{path_save}/res.xlsx", index=True)

ololo = 1