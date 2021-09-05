from pytorch_tabnet.callbacks import Callback
import wandb


def on_train_begin(logs=None):
    metrics_summary = {
        'accuracy_macro': 'max',
        'accuracy_weighted': 'max',
        'f1_macro': 'max',
        'cohen_kappa': 'max',
        'matthews_corrcoef': 'max',
        'f1_weighted': 'max',
    }
    for stage_type in ['train', 'test', 'val']:
        for m, sum in metrics_summary.items():
            wandb.define_metric(f"{stage_type}/{m}", summary=sum)
    wandb.define_metric(f"loss", summary='min')


def on_train_end(logs=None):
    wandb.finish()


def on_epoch_end(epoch, logs=None):
    metrics_summary = {
        'accuracy_macro': 'max',
        'accuracy_weighted': 'max',
        'f1_macro': 'max',
        'cohen_kappa': 'max',
        'matthews_corrcoef': 'max',
        'f1_weighted': 'max',
    }
    for stage_type in ['train', 'test', 'val']:
        log_dict = {}
        for m in metrics_summary:
            log_dict[f"{stage_type}/{m}"] = logs[f"{stage_type}_{m}"]
        log_dict[f"loss"] = logs[f"loss"]
        log_dict[f"epoch"] = epoch
        wandb.log(log_dict)

def set_trainer(self, model=None):
    self.trainer = model


def get_custom_callback():

    TabNetCallback =  type(
        "TabNetCallback",
        (Callback,),
        {
            "set_trainer": set_trainer,
            "on_train_begin": on_train_begin,
            "on_train_end": on_train_end,
            "on_epoch_end": on_epoch_end
        }
    )

    return TabNetCallback
