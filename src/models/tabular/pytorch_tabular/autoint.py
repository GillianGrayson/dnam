from typing import Any, List, Dict
import torch
from src.models.tabular.base import BaseModel
from pytorch_tabular.models.autoint.autoint import AutoIntModel
from omegaconf import DictConfig

class PTAutoIntModel(BaseModel):

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

            continuous_cols,
            categorical_cols,
            attn_embed_dim=32,
            num_heads=2,
            num_attn_blocks=3,
            attn_dropouts=0.0,
            has_residuals=True,
            embedding_dim=16,
            embedding_dropout=0.0,
            deep_layers=False,
            layers="128-64-32",
            activation="ReLU",
            dropout=0.0,
            use_batch_norm=False,
            batch_norm_continuous_input=False,
            attention_pooling=False,
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
                'continuous_cols': self.hparams.continuous_cols,
                'categorical_cols': self.hparams.categorical_cols,
                'continuous_dim': len(self.hparams.continuous_cols),
                'categorical_dim': len(self.hparams.categorical_cols),
                'attn_embed_dim': self.hparams.attn_embed_dim,
                'num_heads': self.hparams.num_heads,
                'num_attn_blocks': self.hparams.num_attn_blocks,
                'attn_dropouts': self.hparams.attn_dropouts,
                'has_residuals': self.hparams.has_residuals,
                'embedding_dim': self.hparams.embedding_dim,
                'embedding_dropout': self.hparams.embedding_dropout,
                'deep_layers': self.hparams.deep_layers,
                'layers': self.hparams.layers,
                'activation': self.hparams.activation,
                'dropout': self.hparams.dropout,
                'use_batch_norm': self.hparams.use_batch_norm,
                'batch_norm_continuous_input': self.hparams.batch_norm_continuous_input,
                'attention_pooling': self.hparams.attention_pooling,
                'initialization': self.hparams.initialization,
            }
        )
        self.model = AutoIntModel(
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
