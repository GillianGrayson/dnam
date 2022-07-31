from typing import Any, List, Dict
import torch
from src.models.tabular.base import BaseModel
from pytorch_tabular.models.tab_transformer.tab_transformer import TabTransformerModel
from omegaconf import DictConfig

class PTTabTransformerModel(BaseModel):

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
            input_embed_dim=32,
            embedding_dropout=0.1,
            share_embedding=False,
            share_embedding_strategy="fraction",
            shared_embedding_fraction=0.25,
            num_heads=8,
            num_attn_blocks=6,
            transformer_head_dim=None,
            attn_dropout=0.1,
            add_norm_dropout=0.1,
            ff_dropout=0.1,
            ff_hidden_multiplier=4,
            transformer_activation="GEGLU",
            out_ff_layers="128-64-32",
            out_ff_activation="ReLU",
            out_ff_dropout=0.0,
            use_batch_norm=False,
            batch_norm_continuous_input=False,
            out_ff_initialization="kaiming",

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
                'input_embed_dim': self.hparams.input_embed_dim,
                'embedding_dropout': self.hparams.embedding_dropout,
                'share_embedding': self.hparams.share_embedding,
                'share_embedding_strategy': self.hparams.share_embedding_strategy,
                'shared_embedding_fraction': self.hparams.shared_embedding_fraction,
                'num_heads': self.hparams.num_heads,
                'num_attn_blocks': self.hparams.num_attn_blocks,
                'transformer_head_dim': self.hparams.transformer_head_dim,
                'attn_dropout': self.hparams.attn_dropout,
                'add_norm_dropout': self.hparams.add_norm_dropout,
                'ff_dropout': self.hparams.ff_dropout,
                'ff_hidden_multiplier': self.hparams.ff_hidden_multiplier,
                'transformer_activation': self.hparams.transformer_activation,
                'out_ff_layers': self.hparams.out_ff_layers,
                'out_ff_activation': self.hparams.out_ff_activation,
                'out_ff_dropout': self.hparams.out_ff_dropout,
                'use_batch_norm': self.hparams.use_batch_norm,
                'batch_norm_continuous_input': self.hparams.batch_norm_continuous_input,
                'out_ff_initialization': self.hparams.out_ff_initialization,
            }
        )
        self.model = TabTransformerModel(
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
