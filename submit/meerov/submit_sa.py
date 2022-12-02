import os


lightgbm_learning_rate = [0.01, 0.1]

args = f"--multirun " \
       f"logger=many_loggers " \
       f"logger.wandb.offline=True " \
       f"experiment=classification/trn_val_tst/sa " \
       f"model_type=lightgbm " \
       f"lightgbm.learning_rate={','.join(str(x) for x in lightgbm_learning_rate)} "

os.system(f"sbatch run_classification_trn_val_tst_sa.sh \"{args}\"")
