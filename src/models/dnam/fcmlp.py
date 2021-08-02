from typing import Any, List
from pytorch_lightning import LightningModule
from torch import nn
import torch
from torchmetrics import Accuracy, F1, Precision, Recall


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
        elif task == "regression":
            if self.hparams.loss_type == "MSE":
                self.loss_fn = torch.nn.MSELoss(reduction='mean')
            elif self.hparams.loss_type == "L1Loss":
                self.loss_fn = torch.nn.L1Loss(reduction='mean')
            else:
                raise ValueError("Unsupported loss_type")

        self.mlp = nn.Sequential(*self.mlp_layers)

        self.accuracy_metric = Accuracy(num_classes=self.n_output)
        self.f1_metric = F1(num_classes=self.n_output)
        self.precision_metric = Precision(num_classes=self.n_output)
        self.recall_metric = Recall(num_classes=self.n_output)

    def forward(self, x: torch.Tensor):
        z = self.mlp(x)
        return z

    def get_probabilities(self, x: torch.Tensor):
        z = self.mlp(x)
        return torch.softmax(z, dim=1)

    def step(self, batch: Any):
        x, y, ind = batch
        out = self.forward(x)
        batch_size = x.size(0)
        if self.task == "regression":
            y = y.view(batch_size, -1)
        loss = self.loss_fn(out, y)

        logs = {"loss": loss}
        non_logs = {}
        if self.task == "classification":
            preds = torch.argmax(out, dim=1)
            non_logs["preds"] = preds
            non_logs["targets"] = y
            logs["acc"] = self.accuracy_metric(preds, y)
            logs["f1"] = self.f1_metric(preds, y)
            logs["precision"] = self.precision_metric(preds, y)
            logs["recall"] = self.recall_metric(preds, y)

        return loss, logs, non_logs

    def training_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch)
        d = {f"train/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch)
        d = {f"val/{k}": v for k, v in logs.items()}
        self.log_dict(d, on_step=False, on_epoch=True, logger=True)
        logs.update(non_logs)
        return logs

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, logs, non_logs = self.step(batch)
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
