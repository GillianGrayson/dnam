from typing import Any, List, Dict
import torch
from src.models.tabular.base import BaseModel
from pytorch_widedeep.models import TabNet


class WDTabNetModel(BaseModel):

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

            column_idx=None,
            cat_embed_input=None,
            cat_embed_dropout=0.1,
            use_cat_bias=False,
            cat_embed_activation=None,
            continuous_cols=None,
            cont_norm_layer='batchnorm',
            embed_continuous=False,
            cont_embed_dim=32,
            cont_embed_dropout=0.1,
            use_cont_bias=True,
            cont_embed_activation=None,
            n_steps=3,
            attn_dim=8,
            dropout=0.0,
            n_glu_step_dependent=2,
            n_glu_shared=2,
            ghost_bn=True,
            virtual_batch_size=128,
            momentum=0.02,
            gamma=1.3,
            epsilon=1e-15,
            mask_type="sparsemax",

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
        self.model = TabNet(
            column_idx=self.hparams.column_idx,
            cat_embed_input=self.hparams.cat_embed_input,
            cat_embed_dropout=self.hparams.cat_embed_dropout,
            use_cat_bias=self.hparams.use_cat_bias,
            cat_embed_activation=self.hparams.cat_embed_activation,
            continuous_cols=self.hparams.continuous_cols,
            cont_norm_layer=self.hparams.cont_norm_layer,
            embed_continuous=self.hparams.embed_continuous,
            cont_embed_dim=self.hparams.cont_embed_dim,
            cont_embed_dropout=self.hparams.cont_embed_dropout,
            use_cont_bias=self.hparams.use_cont_bias,
            cont_embed_activation=self.hparams.cont_embed_activation,
            n_steps=self.hparams.n_steps,
            step_dim=self.hparams.output_dim,
            attn_dim=self.hparams.attn_dim,
            dropout=self.hparams.dropout,
            n_glu_step_dependent=self.hparams.n_glu_step_dependent,
            n_glu_shared=self.hparams.n_glu_shared,
            ghost_bn=self.hparams.ghost_bn,
            virtual_batch_size=self.hparams.virtual_batch_size,
            momentum=self.hparams.momentum,
            gamma=self.hparams.gamma,
            epsilon=self.hparams.epsilon,
            mask_type=self.hparams.mask_type,
        )

    def forward(self, batch: Dict):
        x = batch["all"]
        (x, _) = self.model(x)
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
