from typing import Any, List
from torch import nn
from torchmetrics import MetricCollection, Accuracy, F1, Precision, Recall, CohenKappa, MatthewsCorrcoef, AUROC
import wandb
from typing import Dict
import pytorch_lightning as pl
import torch
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.utils import create_explain_matrix


class TabNetModel(pl.LightningModule):

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
            target_range=None,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self._build_network()

        self.task = task
        self.produce_probabilities = False

        if task == "classification":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            if output_dim < 2:
                raise ValueError(f"Classification with {output_dim} classes")
        elif task == "regression":
            if self.hparams.loss_type == "MSE":
                self.loss_fn = torch.nn.MSELoss(reduction='mean')
            elif self.hparams.loss_type == "L1Loss":
                self.loss_fn = torch.nn.L1Loss(reduction='mean')
            else:
                raise ValueError("Unsupported loss_type")

        self.metrics_dict = {
            'accuracy': Accuracy(num_classes=self.hparams.output_dim),
            'f1_macro': F1(num_classes=self.hparams.output_dim, average='macro'),
            'precision_macro': Precision(num_classes=self.hparams.output_dim, average='macro'),
            'recall_macro': Recall(num_classes=self.hparams.output_dim, average='macro'),
            'f1_weighted': F1(num_classes=self.hparams.output_dim, average='weighted'),
            'precision_weighted': Precision(num_classes=self.hparams.output_dim, average='weighted'),
            'recall_weighted': Recall(num_classes=self.hparams.output_dim, average='weighted'),
            'cohens_kappa': CohenKappa(num_classes=self.hparams.output_dim),
            'matthews_corr': MatthewsCorrcoef(num_classes=self.hparams.output_dim),
        }
        self.metrics_summary = {
            'accuracy': 'max',
            'f1_macro': 'max',
            'precision_macro': 'max',
            'recall_macro': 'max',
            'f1_weighted': 'max',
            'precision_weighted': 'max',
            'recall_weighted': 'max',
            'cohens_kappa': 'max',
            'matthews_corr': 'max',
        }
        self.metrics_prob_dict = {
            'auroc_macro': AUROC(num_classes=self.hparams.output_dim, average='macro'),
            'auroc_weighted': AUROC(num_classes=self.hparams.output_dim, average='weighted'),
        }
        self.metrics_prob_summary = {
            'auroc_macro': 'max',
            'auroc_weighted': 'max',
        }

        self.metrics_train = MetricCollection(self.metrics_dict)
        self.metrics_train_prob = MetricCollection(self.metrics_prob_dict)
        self.metrics_val = self.metrics_train.clone()
        self.metrics_val_prob = self.metrics_train_prob.clone()
        self.metrics_test = self.metrics_train.clone()
        self.metrics_test_prob = self.metrics_train_prob.clone()

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

    def forward(self, x: Dict):
        # Returns output and Masked Loss. We only need the output
        x, _ = self.tabnet(x)
        if (self.hparams.task == "regression") and (
            self.hparams.target_range is not None
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                x[:, i] = y_min + nn.Sigmoid()(x[:, i]) * (y_max - y_min)
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x  # No Easy way to access the raw features in TabNet

    def forward_masks(self, x):
        return self.tabnet.forward_masks(x)

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
