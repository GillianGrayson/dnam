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


class TabTransformerModel(pl.LightningModule):

    def __init__(
            self,
            task="regression",
            input_dim=1,
            output_dim=1,

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

            loss_type="MSE",
            optimizer_lr=0.001,
            optimizer_weight_decay=0.0005,
            scheduler_step_size=20,
            scheduler_gamma=0.9,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        dim_out = output_dim
        self.categories = categories
        self.num_continuous = num_continuous

        self.ids_cat = None
        self.ids_con = None

        self.task = task
        self.produce_probabilities = False

        if task == "classification":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            if output_dim < 2:
                raise ValueError(f"Classification with {output_dim} classes")
            self.metrics_dict = {
                'accuracy_macro': Accuracy(num_classes=self.hparams.output_dim, average='macro'),
                'accuracy_micro': Accuracy(num_classes=self.hparams.output_dim, average='micro'),
                'accuracy_weighted': Accuracy(num_classes=self.hparams.output_dim, average='weighted'),
                'f1_macro': F1(num_classes=self.hparams.output_dim, average='macro'),
                'f1_micro': F1(num_classes=self.hparams.output_dim, average='micro'),
                'f1_weighted': F1(num_classes=self.hparams.output_dim, average='weighted'),
                'precision_macro': Precision(num_classes=self.hparams.output_dim, average='macro'),
                'precision_micro': Precision(num_classes=self.hparams.output_dim, average='micro'),
                'precision_weighted': Precision(num_classes=self.hparams.output_dim, average='weighted'),
                'recall_macro': Recall(num_classes=self.hparams.output_dim, average='macro'),
                'recall_micro': Recall(num_classes=self.hparams.output_dim, average='micro'),
                'recall_weighted': Recall(num_classes=self.hparams.output_dim, average='weighted'),
                'cohens_kappa': CohenKappa(num_classes=self.hparams.output_dim),
                'matthews_corr': MatthewsCorrcoef(num_classes=self.hparams.output_dim),
            }
            self.metrics_summary = {
                'accuracy_macro': 'max',
                'accuracy_micro': 'max',
                'accuracy_weighted': 'max',
                'f1_macro': 'max',
                'f1_micro': 'max',
                'f1_weighted': 'max',
                'precision_macro': 'max',
                'precision_micro': 'max',
                'precision_weighted': 'max',
                'recall_macro': 'max',
                'recall_micro': 'max',
                'recall_weighted': 'max',
                'cohens_kappa': 'max',
                'matthews_corr': 'max',
            }
            self.metrics_prob_dict = {
                'auroc_macro': AUROC(num_classes=self.hparams.output_dim, average='macro'),
                'auroc_micro': AUROC(num_classes=self.hparams.output_dim, average='micro'),
                'auroc_weighted': AUROC(num_classes=self.hparams.output_dim, average='weighted'),
            }
            self.metrics_prob_summary = {
                'auroc_macro': 'max',
                'auroc_micro': 'max',
                'auroc_weighted': 'max',
            }
        elif task == "regression":
            if self.hparams.loss_type == "MSE":
                self.loss_fn = torch.nn.MSELoss(reduction='mean')
            elif self.hparams.loss_type == "L1Loss":
                self.loss_fn = torch.nn.L1Loss(reduction='mean')
            else:
                raise ValueError("Unsupported loss_type")
            self.metrics_dict = {
                'CosineSimilarity': CosineSimilarity(),
                'MeanAbsoluteError': MeanAbsoluteError(),
                'MeanAbsolutePercentageError': MeanAbsolutePercentageError(),
                'MeanSquaredError': MeanSquaredError(),
                'PearsonCorrcoef': PearsonCorrcoef(),
                'R2Score': R2Score(),
                'SpearmanCorrcoef': SpearmanCorrcoef(),
            }
            self.metrics_summary = {
                'CosineSimilarity': 'min',
                'MeanAbsoluteError': 'min',
                'MeanAbsolutePercentageError': 'min',
                'MeanSquaredError': 'min',
                'PearsonCorrcoef': 'max',
                'R2Score': 'max',
                'SpearmanCorrcoef': 'max'
            }
            self.metrics_prob_dict = {}
            self.metrics_prob_summary = {}

        self.metrics_train = MetricCollection(self.metrics_dict)
        self.metrics_train_prob = MetricCollection(self.metrics_prob_dict)
        self.metrics_val = self.metrics_train.clone()
        self.metrics_val_prob = self.metrics_train_prob.clone()
        self.metrics_test = self.metrics_train.clone()
        self.metrics_test_prob = self.metrics_train_prob.clone()

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

        assert x_cont.shape[
                   1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

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
        for stage_type in ['train', 'val', 'test']:
            for m, sum in self.metrics_summary.items():
                wandb.define_metric(f"{stage_type}/{m}", summary=sum)
            for m, sum in self.metrics_prob_summary.items():
                wandb.define_metric(f"{stage_type}/{m}", summary=sum)
            wandb.define_metric(f"{stage_type}/loss", summary='min')

    def step(self, batch: Any, stage:str):
        x, y, ind = batch
        out = self.forward(x)
        batch_size = x.size(0)
        if self.task == "regression":
            y = y.view(batch_size, -1)
        loss = self.loss_fn(out, y)

        logs = {"loss": loss}
        non_logs = {}
        if self.task == "classification":
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)
            non_logs["preds"] = preds
            non_logs["targets"] = y
            if stage == "train":
                logs.update(self.metrics_train(preds, y))
                try:
                    logs.update(self.metrics_train_prob(probs, y))
                except ValueError:
                    pass
            elif stage == "val":
                logs.update(self.metrics_val(preds, y))
                try:
                    logs.update(self.metrics_val_prob(probs, y))
                except ValueError:
                    pass
            elif stage == "test":
                logs.update(self.metrics_test(preds, y))
                try:
                    logs.update(self.metrics_val_prob(probs, y))
                except ValueError:
                    pass
        elif self.task == "regression":
            if stage == "train":
                logs.update(self.metrics_train(out, y))
            elif stage == "val":
                logs.update(self.metrics_val(out, y))
            elif stage == "test":
                logs.update(self.metrics_test(out, y))

        return loss, logs, non_logs

    def training_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "train")
        d = {f"train/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "val")
        d = {f"val/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def predict_step(self, batch, batch_idx):
        x, y, ind = batch
        out = self.forward(x)
        return out

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch, "test")
        d = {f"test/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.optimizer_lr,
            weight_decay=self.hparams.optimizer_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        )
