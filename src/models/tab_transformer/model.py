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
import torch.nn.functional as F
from torch import nn, einsum
from .blocks import *
from einops import rearrange
from src.models.base import BaseModel


class TabTransformerModel(BaseModel):

    def __init__(
            self,
            task="regression",
            loss_type="MSE",
            input_dim=100,
            output_dim=1,
            optimizer_lr=0.001,
            optimizer_weight_decay=0.0005,
            scheduler_step_size=20,
            scheduler_gamma=0.9,

            categories=None,
            num_continuous=None,
            dim=32,
            depth=6,
            heads=8,
            dim_head=16,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=0,
            continuous_mean_std=None,
            attn_dropout=0.1,
            ff_dropout=0.1,

            **kwargs
    ):
        super().__init__(
            task=task,
            loss_type=loss_type,
            input_dim=input_dim,
            output_dim=output_dim,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
        )
        self.save_hyperparameters()

        dim_out = output_dim
        self.categories = categories
        self.num_continuous = num_continuous

        assert all(map(lambda n: n > 0, self.categories)), 'number of each category must be positive'

        # categories related calculations
        self.num_categories = len(self.categories)
        self.num_unique_categories = sum(self.categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(self.categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        # continuous
        if exists(continuous_mean_std):
            assert continuous_mean_std.shape == (self.num_continuous,2), f'continuous_mean_std must have a shape of ({self.num_continuous}, 2) where the last dimension contains the mean and variance respectively'
        self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.norm = nn.LayerNorm(self.num_continuous)

        # transformer
        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        # mlp to logits
        input_size = (dim * self.num_categories) + self.num_continuous
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x):
        x_categ = x[:, self.ids_cat]
        x_cont = x[:, self.ids_con]

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        x_categ += self.categories_offset
        x_categ = x_categ.int()

        x = self.transformer(x_categ)

        flat_categ = x.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if exists(self.continuous_mean_std):
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x_cont = (x_cont - mean) / std

        normed_cont = self.norm(x_cont)

        x = torch.cat((flat_categ, normed_cont), dim=-1)
        x = self.mlp(x)

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
