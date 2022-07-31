from typing import Any, List, Dict
import torch
from src.models.tabular.base import BaseModel
from pytorch_tabular.models.category_embedding.category_embedding_model import CategoryEmbeddingModel
from omegaconf import DictConfig

class PTCategoryEmbeddingModel(BaseModel):

    def __init__(
            self,
            task,
            loss_type,
            input_dim,
            output_dim,
            optimizer_lr,
            optimizer_weight_decay,
            scheduler_step_size,
            scheduler_gamma,

            embedding_dims,
            continuous_cols,
            categorical_cols,
            layers="128-64-32",
            batch_norm_continuous_input=True,
            activation="ReLU",
            embedding_dropout=0.5,
            dropout=0.5,
            use_batch_norm=False,
            initialization="kaiming",

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
        self.save_hyperparameters(logger=False)
        self._build_network()

    def _build_network(self):
        config = DictConfig(
            {
                'task': self.hparams.task,
                'loss': self.hparams.loss_type,
                'metrics': [],
                'metrics_params': [],
                'target_range': None,
                'output_dim': self.hparams.output_dim,
                'embedding_dims': self.hparams.embedding_dims,
                'continuous_cols': self.hparams.continuous_cols,
                'categorical_cols': self.hparams.categorical_cols,
                'continuous_dim': len(self.hparams.continuous_cols),
                'categorical_dim': len(self.hparams.categorical_cols),
                'layers': self.hparams.layers,
                'batch_norm_continuous_input': self.hparams.batch_norm_continuous_input,
                'activation': self.hparams.activation,
                'embedding_dropout': self.hparams.embedding_dropout,
                'dropout': self.hparams.dropout,
                'use_batch_norm': self.hparams.use_batch_norm,
                'initialization': self.hparams.initialization
            }
        )
        self.model = CategoryEmbeddingModel(
            config=config
        )

    def forward(self, batch: Dict):
        x = self.model(batch)['logits']
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x

    def on_train_start(self) -> None:
        super().on_train_start()

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def step(self, batch: Dict, stage:str):
        return super().step(batch=batch, stage=stage)

    def training_step(self, batch: Dict, batch_idx: int):
        return super().training_step(batch=batch, batch_idx=batch_idx)

    def training_epoch_end(self, outputs: List[Any]):
        return super().training_epoch_end(outputs=outputs)

    def validation_step(self, batch: Dict, batch_idx: int):
        return super().validation_step(batch=batch, batch_idx=batch_idx)

    def validation_epoch_end(self, outputs: List[Any]):
        return super().validation_epoch_end(outputs=outputs)

    def test_step(self, batch: Dict, batch_idx: int):
        return super().test_step(batch=batch, batch_idx=batch_idx)

    def test_epoch_end(self, outputs: List[Any]):
        return super().test_epoch_end(outputs=outputs)

    def predict_step(self, batch: Dict, batch_idx):
        return super().predict_step(batch=batch, batch_idx=batch_idx)

    def on_epoch_end(self):
        return super().on_epoch_end()

    def configure_optimizers(self):
        return super().configure_optimizers()
