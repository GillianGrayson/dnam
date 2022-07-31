from typing import Any, List, Dict, Optional, Tuple
import torch
from src.models.tabular.base import BaseModel
from pytorch_widedeep.models import SAINT


class WDSAINTModel(BaseModel):

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

            column_idx: Dict[str, int],
            cat_embed_input: Optional[List[Tuple[str, int]]] = None,
            cat_embed_dropout: float = 0.1,
            use_cat_bias: bool = False,
            cat_embed_activation: Optional[str] = None,
            full_embed_dropout: bool = False,
            shared_embed: bool = False,
            add_shared_embed: bool = False,
            frac_shared_embed: float = 0.25,
            continuous_cols: Optional[List[str]] = None,
            cont_norm_layer: str = None,
            cont_embed_dropout: float = 0.1,
            use_cont_bias: bool = True,
            cont_embed_activation: Optional[str] = None,
            embed_dim: int = 32,
            use_qkv_bias: bool = False,
            n_heads: int = 8,
            n_blocks: int = 2,
            attn_dropout: float = 0.1,
            ff_dropout: float = 0.2,
            transformer_activation: str = "gelu",
            mlp_hidden_dims: Optional[List[int]] = None,
            mlp_activation: str = "relu",
            mlp_dropout: float = 0.1,
            mlp_batchnorm: bool = False,
            mlp_batchnorm_last: bool = False,
            mlp_linear_first: bool = True,

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
        self.model = SAINT(
            column_idx=self.hparams.column_idx,
            cat_embed_input=self.hparams.cat_embed_input,
            cat_embed_dropout=self.hparams.cat_embed_dropout,
            use_cat_bias=self.hparams.use_cat_bias,
            cat_embed_activation=self.hparams.cat_embed_activation,
            full_embed_dropout=self.hparams.full_embed_dropout,
            shared_embed=self.hparams.shared_embed,
            add_shared_embed=self.hparams.add_shared_embed,
            frac_shared_embed=self.hparams.frac_shared_embed,
            continuous_cols=self.hparams.continuous_cols,
            cont_norm_layer=self.hparams.cont_norm_layer,
            cont_embed_dropout=self.hparams.cont_embed_dropout,
            use_cont_bias=self.hparams.use_cont_bias,
            cont_embed_activation=self.hparams.cont_embed_activation,
            input_dim=self.hparams.embed_dim,
            use_qkv_bias=self.hparams.use_qkv_bias,
            n_heads=self.hparams.n_heads,
            n_blocks=self.hparams.n_blocks,
            attn_dropout=self.hparams.attn_dropout,
            ff_dropout=self.hparams.ff_dropout,
            transformer_activation=self.hparams.transformer_activation,
            mlp_hidden_dims=self.hparams.mlp_hidden_dims,
            mlp_activation=self.hparams.mlp_activation,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_batchnorm=self.hparams.mlp_batchnorm,
            mlp_batchnorm_last=self.hparams.mlp_batchnorm_last,
            mlp_linear_first=self.hparams.mlp_linear_first,
        )

    def forward(self, batch: Dict):
        x = batch["all"]
        x = self.model(x)
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
