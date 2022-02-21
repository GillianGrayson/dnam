import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="main.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.utils import utils
    from sa.classification.trn_val_tst.catboost import train_catboost
    from sa.classification.trn_val_tst.xgboost import train_xgboost
    from sa.classification.trn_val_tst.lightgbm import train_lightgbm
    import torch

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('CUDNN VERSION:', torch.backends.cudnn.version())
        print('Number CUDA Devices:', torch.cuda.device_count())
        print('CUDA Device Name:', torch.cuda.get_device_name(0))
        print('CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1024**3)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    if config.sa_model == "catboost":
        return train_catboost(config)
    elif config.sa_model == "xgboost":
        return train_xgboost(config)
    elif config.sa_model == "lightgbm":
        return train_lightgbm(config)
    else:
        raise ValueError(f"Not supported config.sa_model: {config.sa_model}")

if __name__ == "__main__":
    main()