import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    Trainer,
    seed_everything,
)
import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig, NodeConfig


def train(config: DictConfig):

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()

    # Init lightning trainer
    trainer: Trainer = hydra.utils.instantiate(config.trainer, _convert_="partial")

    num_col_names = list(datamodule.betas.columns.values)
    cat_col_names = []
    data = pd.merge(datamodule.pheno.loc[:, datamodule.outcome], datamodule.betas, left_index=True, right_index=True)

    train_data = data.iloc[datamodule.ids_train]
    val_data = data.iloc[datamodule.ids_val]
    test_data = data.iloc[datamodule.ids_test]

    data_config = DataConfig(
        target=[datamodule.outcome],
        # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
    )

    trainer_config = TrainerConfig(
        auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
        batch_size=datamodule.batch_size,
        max_epochs=trainer.max_epochs,
        gpus=-1,  # index of the GPU to use. -1 means all available GPUs, None, means CPU
    )

    optimizer_config = OptimizerConfig()

    model_config = NodeConfig(
        task="classification",
        num_layers=2,  # Number of Dense Layers
        num_trees=512,  # Number of Trees in each layer
        depth=5,  # Depth of each Tree
        embed_categorical=False, # If True, will use a learned embedding, else it will use LeaveOneOutEncoding for categorical columns
        learning_rate=1e-3,
        metrics=["accuracy", "f1", 'precision', 'recall'],
        metrics_params = [{}, {'num_classes': config.model.n_output, 'average': 'weighted'}, {'num_classes': config.model.n_output, 'average': 'weighted'}, {'num_classes': config.model.n_output, 'average': 'weighted'}]
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    tabular_model.fit(train=train_data, validation=val_data)

    result = tabular_model.evaluate(test_data)
    pred_df = tabular_model.predict(test_data)
    pred_df.head()
    ololo = 1


