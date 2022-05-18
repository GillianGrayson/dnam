import os

model_type = 'lightgbm'

catboost_learning_rate = [0.05]
catboost_depth = [4]
catboost_min_data_in_leaf = [1]
catboost_max_leaves = [31]

lightgbm_learning_rate = [0.01]
lightgbm_num_leaves = [31]
lightgbm_min_data_in_leaf = [20]
lightgbm_feature_fraction = [0.9]
lightgbm_bagging_fraction = [0.8]

xgboost_learning_rate = [0.01]
xgboost_booster = ['gbtree']
xgboost_max_depth = [6]
xgboost_gamma = [0]
xgboost_subsample = [1.0]

base_dir = "/common/home/yusipov_i/data/dnam/datasets/meta/GPL13534_Blood/Schizophrenia"
in_dim = 200

args = f"--multirun " \
       f"logger=many_loggers " \
       f"logger.wandb.offline=True " \
       f"base_dir={base_dir} " \
       f"model_type={model_type} " \
       f"in_dim={in_dim} " \
       f"experiment=dnam/multiclass/trn_val_tst/sa "

if model_type == 'catboost':
    args += f"catboost.learning_rate={','.join(str(x) for x in catboost_learning_rate)} " \
            f"catboost.depth={','.join(str(x) for x in catboost_depth)} " \
            f"catboost.min_data_in_leaf={','.join(str(x) for x in catboost_min_data_in_leaf)} " \
            f"catboost.max_leaves={','.join(str(x) for x in catboost_max_leaves)} "
elif model_type == 'lightgbm':
    args += f"lightgbm.learning_rate={','.join(str(x) for x in lightgbm_learning_rate)} " \
            f"lightgbm.num_leaves={','.join(str(x) for x in lightgbm_num_leaves)} " \
            f"lightgbm.min_data_in_leaf={','.join(str(x) for x in lightgbm_min_data_in_leaf)} " \
            f"lightgbm.feature_fraction={','.join(str(x) for x in lightgbm_feature_fraction)} " \
            f"lightgbm.bagging_fraction={','.join(str(x) for x in lightgbm_bagging_fraction)} "
elif model_type == 'xgboost':
    args += f"xgboost.learning_rate={','.join(str(x) for x in xgboost_learning_rate)} " \
            f"xgboost.booster={','.join(str(x) for x in xgboost_booster)} " \
            f"xgboost.max_depth={','.join(str(x) for x in xgboost_max_depth)} " \
            f"xgboost.gamma={','.join(str(x) for x in xgboost_gamma)} " \
            f"xgboost.subsample={','.join(str(x) for x in xgboost_subsample)} "
else:
    raise ValueError(f"Unsupported model_type: {model_type}")

os.system(f"sbatch run_multiclass_trn_val_tst_sa.sh \"{args}\"")
