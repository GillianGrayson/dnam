from typing import Any, List, Dict
from pytorch_lightning import LightningModule
from torch import nn
import torch
from torchmetrics import MetricCollection, Accuracy, F1, Precision, Recall, AUROC, CohenKappa, MatthewsCorrcoef, ConfusionMatrix
from torchmetrics import CosineSimilarity, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, PearsonCorrcoef, R2Score, SpearmanCorrcoef
import wandb


class FCMLPModel(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            n_input: int = 10,
            topology: List[int] = None,
            n_output: int = 1,
            task: str = "regression",
            dropout: float = 0.1,
            loss_type: str = "MSE",
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.n_output = n_output
        self.topology = [n_input] + list(topology) + [n_output]

        self.mlp_layers = []
        for i in range(len(self.topology) - 2):
            layer = nn.Linear(self.topology[i], self.topology[i + 1])
            self.mlp_layers.append(nn.Sequential(layer, nn.LeakyReLU(), nn.Dropout(dropout)))
        output_layer = nn.Linear(self.topology[-2], self.topology[-1])
        self.mlp_layers.append(output_layer)


        if task == "classification":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            if n_output < 2:
                raise ValueError(f"Classification with {n_output} classes")
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
                'cohen_kappa': CohenKappa(num_classes=self.hparams.output_dim),
                'matthews_corr_coef': MatthewsCorrcoef(num_classes=self.hparams.output_dim),
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

    def forward(self, x: torch.Tensor):
        z = self.mlp(x)
        return z

    def get_probabilities(self, x: torch.Tensor):
        z = self.mlp(x)
        return torch.softmax(z, dim=1)

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
                logs.update(self.metrics_train_prob(probs, y))
            elif stage == "val":
                logs.update(self.metrics_val(preds, y))
                logs.update(self.metrics_val_prob(probs, y))
            elif stage == "test":
                logs.update(self.metrics_test(preds, y))
                logs.update(self.metrics_test_prob(probs, y))

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
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
