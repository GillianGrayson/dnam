import os

model_sa = 'catboost'

catboost_learning_rate = [0.01]
catboost_depth = [6]
catboost_min_data_in_leaf = [2]
catboost_max_leaves = [31]

base_dir = "/common/home/yusipov_i/data/dnam/datasets/meta/GPL13534_Blood/Schizophrenia"
in_dim = 200

if model_sa == 'catboost':
    args = f"--multirun " \
           f"logger=csv " \
           f"base_dir={base_dir} " \
           f"model_sa={model_sa} " \
           f"in_dim={in_dim} " \
           f"experiment=dnam/multiclass/trn_val_tst/sa " \
           f"catboost.learning_rate={','.join(str(x) for x in catboost_learning_rate)} " \
           f"catboost.depth={','.join(str(x) for x in catboost_depth)} " \
           f"catboost.min_data_in_leaf={','.join(str(x) for x in catboost_min_data_in_leaf)} " \
           f"catboost.max_leaves={','.join(str(x) for x in catboost_max_leaves)} "

os.system(f"sbatch run_multiclass_trn_val_tst_sa.sh \"{args}\"")
