import os
from pathlib import Path
import numpy as np
import pandas as pd


model_sa = 'xgboost'

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
feat_imp_fn = f"{base_dir}/harmonized/models/baseline/dnam_harmonized_multiclass_Status_trn_val_tst_{model_sa}/runs/2022-03-25_23-14-14/feature_importances.xlsx"
feat_imp_df = pd.read_excel(feat_imp_fn, index_col="feature")
feat_imp_df.sort_values(['importance'], ascending=[False], inplace=True)
cpgs_path = f"{base_dir}/harmonized/cpgs/serial"
Path(cpgs_path).mkdir(parents=True, exist_ok=True)
# n_feats = np.linspace(10, 500, 50, dtype=int)
n_feats = [10]

for n_feat in n_feats:
    project_name = f'dnam_harmonized_multiclass_Status_trn_val_tst_{model_sa}_n_feat'
    feats_df = feat_imp_df.head(n_feat)
    feats_df.to_excel(f"{cpgs_path}/{n_feat}.xlsx", index=True)
    features_fn = f"{cpgs_path}/{n_feat}.xlsx"

    args = f"--multirun " \
           f"project_name={project_name} " \
           f"logger=many_loggers " \
           f"logger.wandb.offline=True " \
           f"base_dir={base_dir} " \
           f"model_sa={model_sa} " \
           f"in_dim={n_feat} " \
           f"datamodule.features_fn={features_fn} " \
           f"experiment=dnam/multiclass/trn_val_tst/sa "

    if model_sa == 'catboost':
        args += f"catboost.learning_rate={','.join(str(x) for x in catboost_learning_rate)} " \
                f"catboost.depth={','.join(str(x) for x in catboost_depth)} " \
                f"catboost.min_data_in_leaf={','.join(str(x) for x in catboost_min_data_in_leaf)} " \
                f"catboost.max_leaves={','.join(str(x) for x in catboost_max_leaves)} "
    elif model_sa == 'lightgbm':
        args += f"lightgbm.learning_rate={','.join(str(x) for x in lightgbm_learning_rate)} " \
                f"lightgbm.num_leaves={','.join(str(x) for x in lightgbm_num_leaves)} " \
                f"lightgbm.min_data_in_leaf={','.join(str(x) for x in lightgbm_min_data_in_leaf)} " \
                f"lightgbm.feature_fraction={','.join(str(x) for x in lightgbm_feature_fraction)} " \
                f"lightgbm.bagging_fraction={','.join(str(x) for x in lightgbm_bagging_fraction)} "
    elif model_sa == 'xgboost':
        args += f"xgboost.learning_rate={','.join(str(x) for x in xgboost_learning_rate)} " \
                f"xgboost.booster={','.join(str(x) for x in xgboost_booster)} " \
                f"xgboost.max_depth={','.join(str(x) for x in xgboost_max_depth)} " \
                f"xgboost.gamma={','.join(str(x) for x in xgboost_gamma)} " \
                f"xgboost.subsample={','.join(str(x) for x in xgboost_subsample)} "
    else:
        raise ValueError(f"Unsupported model_sa: {model_sa}")

    os.system(f"sbatch run_multiclass_trn_val_tst_sa.sh \"{args}\"")
