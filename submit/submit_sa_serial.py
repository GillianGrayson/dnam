import os
from pathlib import Path
import numpy as np
import pandas as pd

disease = "Schizophrenia"
data_type = "harmonized"
model_sa = 'catboost'
run_type = "trn_val_tst"

catboost_learning_rate = [0.05, 0.01]
catboost_depth = [4, 6]
catboost_min_data_in_leaf = [1, 4]
catboost_max_leaves = [31]

lightgbm_learning_rate = [0.01, 0.05]
lightgbm_num_leaves = [31, 63]
lightgbm_min_data_in_leaf = [5, 10]
lightgbm_feature_fraction = [0.9]
lightgbm_bagging_fraction = [0.8]

xgboost_learning_rate = [0.01, 0.05]
xgboost_booster = ['gbtree']
xgboost_max_depth = [6, 8]
xgboost_gamma = [0]
xgboost_subsample = [1.0, 0.5]

base_dir = f"/common/home/yusipov_i/data/dnam/datasets/meta/GPL13534_Blood/{disease}"
feat_imp_fn = f"{base_dir}/{data_type}/models/baseline/{disease}_{data_type}_{run_type}_{model_sa}/runs/2022-03-31_00-58-59/feature_importances.xlsx"

feat_imp_df = pd.read_excel(feat_imp_fn, index_col="feature")
feat_imp_df.index.name = "features"
feat_imp_df.sort_values(['importance'], ascending=[False], inplace=True)
cpgs_path = f"{base_dir}/{data_type}/cpgs/serial/{run_type}/{model_sa}"
Path(cpgs_path).mkdir(parents=True, exist_ok=True)
n_feats = np.linspace(10, 1000, 100, dtype=int)
# n_feats = [10]

for n_feat in n_feats:
    project_name = f'{disease}_{data_type}_{run_type}_{model_sa}_{n_feat}'
    feats_df = feat_imp_df.head(n_feat)
    feats_df.to_excel(f"{cpgs_path}/{n_feat}.xlsx", index=True)
    features_fn = f"{cpgs_path}/{n_feat}.xlsx"

    args = f"--multirun " \
           f"disease={disease} " \
           f"data_type={data_type} " \
           f"model_sa={model_sa} " \
           f"project_name={project_name} " \
           f"logger=many_loggers " \
           f"logger.wandb.offline=True " \
           f"base_dir={base_dir} " \
           f"in_dim={n_feat} " \
           f"datamodule.features_fn={features_fn} " \
           f"experiment=dnam/multiclass/{run_type}/sa "

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
