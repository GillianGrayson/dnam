import os

model_sa = 'catboost'

catboost_learning_rate = [0.05, 0.005]
catboost_depth = [6, 8]
catboost_min_data_in_leaf = [1, 2, 5]
catboost_max_leaves = [31, 63]

if model_sa == 'catboost':
    args = f"--multirun " \
           f"model_sa={model_sa} " \
           f"logger.wandb.offline=True " \
           f"experiment=unn/dnam/multiclass/trn_val_tst/sa " \
           f"catboost.learning_rate={','.join(str(x) for x in catboost_learning_rate)} " \
           f"catboost.depth={','.join(str(x) for x in catboost_depth)} " \
           f"catboost.min_data_in_leaf={','.join(str(x) for x in catboost_min_data_in_leaf)} " \
           f"catboost.max_leaves={','.join(str(x) for x in catboost_max_leaves)} "

os.system(f"sbatch run_multiclass_trn_val_tst_sa.sh \"{args}\"")
