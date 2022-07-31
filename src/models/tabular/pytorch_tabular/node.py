from typing import Any, List, Dict
import torch
from src.models.tabular.base import BaseModel
from pytorch_tabular.models.node.node_model import NODEModel
from omegaconf import DictConfig

class PTNODEModel(BaseModel):

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
            num_layers=1,
            num_trees=2048,
            additional_tree_output_dim=3,
            depth=6,
            choice_function="entmax15",
            bin_function="entmoid15",
            max_features=None,
            input_dropout=0.0,
            initialize_response="normal",
            initialize_selection_logits="uniform",
            threshold_init_beta=1.0,
            threshold_init_cutoff=1.0,
            embed_categorical=False,
            embedding_dropout=0.0,

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
                'num_layers': self.hparams.num_layers,
                'num_trees': self.hparams.num_trees,
                'additional_tree_output_dim': self.hparams.additional_tree_output_dim,
                'depth': self.hparams.depth,
                'choice_function': self.hparams.choice_function,
                'bin_function': self.hparams.bin_function,
                'max_features': self.hparams.max_features,
                'input_dropout': self.hparams.input_dropout,
                'initialize_response': self.hparams.initialize_response,
                'initialize_selection_logits': self.hparams.initialize_selection_logits,
                'threshold_init_beta': self.hparams.threshold_init_beta,
                'threshold_init_cutoff': self.hparams.threshold_init_cutoff,
                'embed_categorical': self.hparams.embed_categorical,
                'embedding_dropout': self.hparams.embedding_dropout,
            }
        )
        self.model = NODEModel(
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
