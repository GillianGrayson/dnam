import os

import numpy as np

data_type = "immuno"
model_type = "lightgbm"
run_type = "trn_val"

if model_type in ["tabnet", "node", "tab_transformer"]:
    sa_or_pl = "pl"
else:
    sa_or_pl = "sa"

features_fn = "features.xlsx"
trn_val_fn = "data_thld_25.xlsx"

catboost_learning_rate = [0.1, 0.05, 0.01, 0.005]
catboost_depth = [3, 4, 5]
catboost_min_data_in_leaf = list(np.linspace(1, 20, 20, dtype=int))
catboost_max_leaves = [15, 31]

lightgbm_learning_rate = [0.05, 0.01, 0.005, 0.001]
lightgbm_num_leaves = [15, 31]
lightgbm_min_data_in_leaf = list(np.linspace(1, 20, 20, dtype=int))
lightgbm_feature_fraction = [0.9, 0.8]
lightgbm_bagging_fraction = [0.8, 0.7]

xgboost_learning_rate = [0.1, 0.05, 0.01, 0.005]
xgboost_booster = ['gbtree']
xgboost_max_depth = [4, 5, 6, 7]
xgboost_gamma = [0]
xgboost_subsample = [1.0]

elastic_net_alpha = list(np.logspace(-3, 2, 101))
elastic_net_l1_ratio = [0.5]
elastic_net_max_iter = [100000]
elastic_net_tol = [1e-2]

tabnet_n_d_n_a = [8]
tabnet_n_steps = [3]
tabnet_gamma = [1.3]
tabnet_n_independent = [1]
tabnet_n_shared = [2]
tabnet_optimizer_lr = list(np.logspace(-4, 0, 15))
tabnet_optimizer_weight_decay = [0.0]
tabnet_scheduler_step_size = [50]
tabnet_scheduler_gamma = [0.9]

node_num_trees = [1024, 512]
node_num_layers = [1]
node_depth = [5, 4]
node_optimizer_lr = list(np.logspace(-2, 0, 6))
node_optimizer_weight_decay = [0.0]
node_scheduler_step_size = [50]
node_scheduler_gamma = [0.9]

tab_transformer_dim = [32]
tab_transformer_depth = [6]
tab_transformer_heads = [8]
tab_transformer_dim_head = [16]
tab_transformer_num_special_tokens = [0]
tab_transformer_attn_dropout = [0.0]
tab_transformer_ff_dropout = [0.0]
tab_transformer_optimizer_lr = list(np.logspace(-4, 0, 15))
tab_transformer_optimizer_weight_decay = [0.0]
tab_transformer_scheduler_step_size = [30]
tab_transformer_scheduler_gamma = [0.9]

base_dir = f"/common/home/yusipov_i/data/unn/{data_type}"

args = f"--multirun " \
       f"logger=many_loggers " \
       f"logger.wandb.offline=True " \
       f"base_dir={base_dir} " \
       f"model_type={model_type} " \
       f"datamodule.features_fn={base_dir}/{features_fn} " \
       f"datamodule.trn_val_fn={base_dir}/{trn_val_fn} " \
       f"experiment=unn/regression/{run_type}/{sa_or_pl} "

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
elif model_type == 'elastic_net':
    args += f"elastic_net.alpha={','.join(str(x) for x in elastic_net_alpha)} " \
            f"elastic_net.l1_ratio={','.join(str(x) for x in elastic_net_l1_ratio)} " \
            f"elastic_net.max_iter={','.join(str(x) for x in elastic_net_max_iter)} " \
            f"elastic_net.tol={','.join(str(x) for x in elastic_net_tol)} "
elif model_type == 'tabnet':
    args += f"tabnet.n_d_n_a={','.join(str(x) for x in tabnet_n_d_n_a)} " \
            f"tabnet.n_steps={','.join(str(x) for x in tabnet_n_steps)} " \
            f"tabnet.gamma={','.join(str(x) for x in tabnet_gamma)} " \
            f"tabnet.n_independent={','.join(str(x) for x in tabnet_n_independent)} " \
            f"tabnet.n_shared={','.join(str(x) for x in tabnet_n_shared)} " \
            f"tabnet.optimizer_lr={','.join(str(x) for x in tabnet_optimizer_lr)} " \
            f"tabnet.optimizer_weight_decay={','.join(str(x) for x in tabnet_optimizer_weight_decay)} " \
            f"tabnet.scheduler_step_size={','.join(str(x) for x in tabnet_scheduler_step_size)} " \
            f"tabnet.scheduler_gamma={','.join(str(x) for x in tabnet_scheduler_gamma)} " \
            f"trainer.gpus=0 "
elif model_type == 'node':
    args += f"node.num_trees={','.join(str(x) for x in node_num_trees)} " \
            f"node.num_layers={','.join(str(x) for x in node_num_layers)} " \
            f"node.depth={','.join(str(x) for x in node_depth)} " \
            f"node.optimizer_lr={','.join(str(x) for x in node_optimizer_lr)} " \
            f"node.optimizer_weight_decay={','.join(str(x) for x in node_optimizer_weight_decay)} " \
            f"node.scheduler_step_size={','.join(str(x) for x in node_scheduler_step_size)} " \
            f"node.scheduler_gamma={','.join(str(x) for x in node_scheduler_gamma)} " \
            f"trainer.gpus=0 "
elif model_type == 'tab_transformer':
    args += f"tab_transformer.dim={','.join(str(x) for x in tab_transformer_dim)} " \
            f"tab_transformer.depth={','.join(str(x) for x in tab_transformer_depth)} " \
            f"tab_transformer.heads={','.join(str(x) for x in tab_transformer_heads)} " \
            f"tab_transformer.dim_head={','.join(str(x) for x in tab_transformer_dim_head)} " \
            f"tab_transformer.num_special_tokens={','.join(str(x) for x in tab_transformer_num_special_tokens)} " \
            f"tab_transformer.attn_dropout={','.join(str(x) for x in tab_transformer_attn_dropout)} " \
            f"tab_transformer.ff_dropout={','.join(str(x) for x in tab_transformer_ff_dropout)} " \
            f"tab_transformer.optimizer_lr={','.join(str(x) for x in tab_transformer_optimizer_lr)} " \
            f"tab_transformer.optimizer_weight_decay={','.join(str(x) for x in tab_transformer_optimizer_weight_decay)} "\
            f"tab_transformer.scheduler_step_size={','.join(str(x) for x in tab_transformer_scheduler_step_size)} " \
            f"tab_transformer.scheduler_gamma={','.join(str(x) for x in tab_transformer_scheduler_gamma)} " \
            f"trainer.gpus=0 "
else:
    raise ValueError(f"Unsupported model_type: {model_type}")

os.system(f"sbatch run_regression_trn_val_tst_{sa_or_pl}.sh \"{args}\"")
