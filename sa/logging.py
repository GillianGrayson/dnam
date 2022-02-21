from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
        loggers,
        config: DictConfig
) -> None:

    for logger in loggers:
        logger.log_hyperparams(config)

        # disable logging any more hyperparameters for all loggers
        # this is just a trick to prevent trainer from logging hparams of model,
        # since we already did that above
        logger.log_hyperparams = empty

