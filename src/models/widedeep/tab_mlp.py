from typing import Any, List
import torch
from pytorch_tabnet.tab_network import TabNet
from src.models.base import BaseModel
from pytorch_widedeep.models import TabMlp


class TabMLPModel(BaseModel):

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

            features,

            cat_embed_input=None,
            cat_embed_dropout=0.1,
            use_cat_bias=False,
            cat_embed_activation=None,
            cont_norm_layer='batchnorm',
            embed_continuous=False,
            cont_embed_dim=32,
            cont_embed_dropout=0.1,
            use_cont_bias=True,
            cont_embed_activation=None,
            mlp_hidden_dims=(200, 100),
            mlp_activation='relu',
            mlp_dropout=0.1,
            mlp_batchnorm=False,
            mlp_batchnorm_last=False,
            mlp_linear_first=False,

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
        self.save_hyperparameters(ignore='column_idx')
        features = list(features)
        self.column_idx = {k:v for v,k in enumerate(features)}
        self.continuous_cols = features
        self._build_network()

    def _build_network(self):
        self.model = TabMlp(
            column_idx=self.column_idx,
            cat_embed_input=self.hparams.cat_embed_input,
            cat_embed_dropout=self.hparams.cat_embed_dropout,
            use_cat_bias=self.hparams.use_cat_bias,
            cat_embed_activation=self.hparams.cat_embed_activation,
            continuous_cols=self.continuous_cols,
            cont_norm_layer=self.hparams.cont_norm_layer,
            embed_continuous=self.hparams.embed_continuous,
            cont_embed_dim=self.hparams.cont_embed_dim,
            cont_embed_dropout=self.hparams.cont_embed_dropout,
            use_cont_bias=self.hparams.use_cont_bias,
            cont_embed_activation=self.hparams.cont_embed_activation,
            mlp_hidden_dims=self.hparams.mlp_hidden_dims,
            mlp_activation=self.hparams.mlp_activation,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_batchnorm=self.hparams.mlp_batchnorm,
            mlp_batchnorm_last=self.hparams.mlp_batchnorm_last,
            mlp_linear_first=self.hparams.mlp_linear_first
        )

    def forward(self, x):
        x = self.model(x)
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x

    def on_train_start(self) -> None:
        super().on_train_start()

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

    def on_epoch_end(self):
        return super().on_epoch_end()

    def configure_optimizers(self):
        return super().configure_optimizers()
