from typing import Any, List
from torch import nn
from torchmetrics import MetricCollection, Accuracy, F1, Precision, Recall, CohenKappa, MatthewsCorrcoef, AUROC
from torchmetrics import CosineSimilarity, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, PearsonCorrcoef, R2Score, SpearmanCorrcoef
import wandb
from typing import Dict
import pytorch_lightning as pl
import torch
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.utils import create_explain_matrix
from src.models.base import BaseModel


class TabNetModel(BaseModel):

    def __init__(
            self,
            task="classification",
            input_dim=100,
            output_dim=4,

            n_d_n_a=8,
            n_steps=3,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            virtual_batch_size=128,
            mask_type="sparsemax",

            loss_type="MSE",
            optimizer_lr=0.001,
            optimizer_weight_decay=0.0005,
            scheduler_step_size=20,
            scheduler_gamma=0.9,
            **kwargs
    ):
        super().__init__(task=task, output_dim=output_dim)
        self.save_hyperparameters()
        self._build_network()

    def _build_network(self):
        self.tabnet = TabNet(
            input_dim=self.hparams.input_dim,
            output_dim=self.hparams.output_dim,
            n_d=self.hparams.n_d_n_a,
            n_a=self.hparams.n_d_n_a,
            n_steps=self.hparams.n_steps,
            gamma=self.hparams.gamma,
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=[],
            n_independent=self.hparams.n_independent,
            n_shared=self.hparams.n_shared,
            epsilon=1e-15,
            virtual_batch_size=self.hparams.virtual_batch_size,
            momentum=0.02,
            mask_type=self.hparams.mask_type,
        )

    def forward(self, x):
        # Returns output and Masked Loss. We only need the output
        if self.produce_importance:
            return self.tabnet.forward_masks(x)
        else:
            x, _ = self.tabnet(x)
            if self.produce_probabilities:
                return torch.softmax(x, dim=1)
            else:
                return x

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def step(self, batch: Any, stage:str):
        return super().step(batch=batch, stage=stage)

    def training_step(self, batch: Any, batch_idx: int):
        return super().training_step(batch=batch, batch_idx=batch_idx)

    def training_epoch_end(self, outputs: List[Any]):
        return super().training_epoch_end(outputs=outputs)

    def validation_step(self, batch: Any, batch_idx: int):
        return super().validation_step(batch=batch, batch_idx=batch_idx)

    def validation_epoch_end(self, outputs: List[Any]):
        return super().validation_epoch_end(outputs=outputs)

    def test_step(self, batch: Any, batch_idx: int):
        return super().test_step(batch=batch, batch_idx=batch_idx)

    def test_epoch_end(self, outputs: List[Any]):
        return super().test_epoch_end(outputs=outputs)

    def predict_step(self, batch, batch_idx):
        return super().predict_step(batch=batch, batch_idx=batch_idx)

    def configure_optimizers(self):
        return super().configure_optimizers()
