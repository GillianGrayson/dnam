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


class BaseModel(pl.LightningModule):

    def __init__(
            self,
            task,
            input_dim,
            output_dim,
            loss_type,
            optimizer_lr,
            optimizer_weight_decay,
            scheduler_step_size,
            scheduler_gamma,
    ):
        super().__init__()

        self.task = task
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_type = loss_type
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        self.ids_cat = None
        self.ids_con = None

        self.produce_probabilities = False
        self.produce_importance = False

        if self.task == "classification":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            if self.output_dim < 2:
                raise ValueError(f"Classification with {self.output_dim} classes")
            self.metrics_dict = {
                'accuracy_macro': Accuracy(num_classes=self.output_dim, average='macro'),
                'accuracy_micro': Accuracy(num_classes=self.output_dim, average='micro'),
                'accuracy_weighted': Accuracy(num_classes=self.output_dim, average='weighted'),
                'f1_macro': F1(num_classes=self.output_dim, average='macro'),
                'f1_micro': F1(num_classes=self.output_dim, average='micro'),
                'f1_weighted': F1(num_classes=self.output_dim, average='weighted'),
                'precision_macro': Precision(num_classes=self.output_dim, average='macro'),
                'precision_micro': Precision(num_classes=self.output_dim, average='micro'),
                'precision_weighted': Precision(num_classes=self.output_dim, average='weighted'),
                'recall_macro': Recall(num_classes=self.output_dim, average='macro'),
                'recall_micro': Recall(num_classes=self.output_dim, average='micro'),
                'recall_weighted': Recall(num_classes=self.output_dim, average='weighted'),
                'cohen_kappa': CohenKappa(num_classes=self.output_dim),
                'matthews_corr_coef': MatthewsCorrcoef(num_classes=self.output_dim),
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
                'cohen_kappa': 'max',
                'matthews_corr_coef': 'max',
            }
            self.metrics_prob_dict = {
                'auroc_macro': AUROC(num_classes=self.output_dim, average='macro'),
                'auroc_micro': AUROC(num_classes=self.output_dim, average='micro'),
                'auroc_weighted': AUROC(num_classes=self.output_dim, average='weighted'),
            }
            self.metrics_prob_summary = {
                'auroc_macro': 'max',
                'auroc_micro': 'max',
                'auroc_weighted': 'max',
            }
        elif self.task == "regression":
            if self.loss_type == "MSE":
                self.loss_fn = torch.nn.MSELoss(reduction='mean')
            elif self.loss_type == "L1Loss":
                self.loss_fn = torch.nn.L1Loss(reduction='mean')
            else:
                raise ValueError("Unsupported loss_type")
            self.metrics_dict = {
                'cosine_similarity': CosineSimilarity(),
                'mean_absolute_error': MeanAbsoluteError(),
                'mean_absolute_percentage_error': MeanAbsolutePercentageError(),
                'mean_squared_error': MeanSquaredError(),
                'pearson_corr_coef': PearsonCorrcoef(),
                'r2_score': R2Score(),
                'spearman_corr_coef': SpearmanCorrcoef(),
            }
            self.metrics_summary = {
                'cosine_similarity': 'min',
                'mean_absolute_error': 'min',
                'mean_absolute_percentage_error': 'min',
                'mean_squared_error': 'min',
                'pearson_corr_coef': 'max',
                'r2_score': 'max',
                'spearman_corr_coef': 'max'
            }
            self.metrics_prob_dict = {}
            self.metrics_prob_summary = {}

        self.metrics_train = MetricCollection(self.metrics_dict)
        self.metrics_train_prob = MetricCollection(self.metrics_prob_dict)
        self.metrics_val = self.metrics_train.clone()
        self.metrics_val_prob = self.metrics_train_prob.clone()
        self.metrics_test = self.metrics_train.clone()
        self.metrics_test_prob = self.metrics_train_prob.clone()

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

    def predict_step(self, batch, batch_idx):
        x, y, ind = batch
        out = self.forward(x)
        return out

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        )
