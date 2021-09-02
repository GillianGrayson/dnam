from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import random
import numpy as np
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



def make_mixed_classification(n_samples, n_features, n_categories):
    X,y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42, n_informative=5)
    cat_cols = random.choices(list(range(X.shape[-1])),k=n_categories)
    num_cols = [i for i in range(X.shape[-1]) if i not in cat_cols]
    for col in cat_cols:
        X[:,col] = pd.qcut(X[:,col], q=4).codes.astype(int)
    col_names = []
    num_col_names=[]
    cat_col_names=[]
    for i in range(X.shape[-1]):
        if i in cat_cols:
            col_names.append(f"cat_col_{i}")
            cat_col_names.append(f"cat_col_{i}")
        if i in num_cols:
            col_names.append(f"num_col_{i}")
            num_col_names.append(f"num_col_{i}")
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y, name="target")
    data = X.join(y)
    return data, cat_col_names, num_col_names

def print_metrics(y_true, y_pred, tag):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_true.ndim>1:
        y_true=y_true.ravel()
    if y_pred.ndim>1:
        y_pred=y_pred.ravel()
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred)
    print(f"{tag} Acc: {val_acc} | {tag} F1: {val_f1}")


@hydra.main(config_path="../../configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

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

    train = data.iloc[datamodule.ids_train]
    val = data.iloc[datamodule.ids_val]
    test = data.iloc[datamodule.ids_test]

    # data, cat_col_names, num_col_names = make_mixed_classification(n_samples=10000, n_features=20, n_categories=4)
    # train, test = train_test_split(data, random_state=42)
    # train, val = train_test_split(train, random_state=42)

    data_config = DataConfig(
        target=['target'],
        # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
    )
    trainer_config = TrainerConfig(
        auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
        batch_size=30,
        max_epochs=3000,
        gpus=-1,  # index of the GPU to use. -1 means all available GPUs, None, means CPU
    )
    optimizer_config = OptimizerConfig()

    # model_config = CategoryEmbeddingModelConfig(
    #     task="classification",
    #     layers="1024-512-512",  # Number of nodes in each layer
    #     activation="LeakyReLU", # Activation between each layers
    #     learning_rate = 1e-3
    # )

    model_config = NodeConfig(
        task="classification",
        num_layers=2,  # Number of Dense Layers
        num_trees=512,  # Number of Trees in each layer
        depth=5,  # Depth of each Tree
        embed_categorical=False,
        # If True, will use a learned embedding, else it will use LeaveOneOutEncoding for categorical columns
        learning_rate=1e-3
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    tabular_model.fit(train=train, validation=val)

    result = tabular_model.evaluate(test)

    pred_df = tabular_model.predict(test)
    pred_df.head()


if __name__ == "__main__":
    main()