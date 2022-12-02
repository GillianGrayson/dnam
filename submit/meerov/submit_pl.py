import os


tabnet_optimizer_lr = [0.1, 0.005]

args = f"--multirun " \
       f"logger=many_loggers " \
       f"logger.wandb.offline=True " \
       f"experiment=classification/trn_val_tst/pl " \
       f"model_type=tabnet " \
       f"tabnet.optimizer_lr={','.join(str(x) for x in tabnet_optimizer_lr)} " \
       f"trainer.gpus=0 "

os.system(f"sbatch run_classification_trn_val_tst_pl.sh \"{args}\"")
