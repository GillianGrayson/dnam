import os

segment = 'a100'

base_dir = f"/common/home/yusipov_i/data/unn/immuno"

max_epochs = 1000
patience = 100
progress_bar_refresh_rate = 0

model_dict = {
    'pytorch_tabular_node': ('pytorch_tabular_node', 'pytorch'),
    'danet': ('danet', 'pytorch'),
    'nam': ('nam', 'pytorch'),
    'widedeep_tab_net': ('widedeep_tab_net', 'pytorch'),
    'widedeep_saint': ('widedeep_saint', 'pytorch'),
    'widedeep_ft_transformer': ('widedeep_ft_transformer', 'pytorch'),
    'pytorch_tabular_autoint': ('pytorch_tabular_autoint', 'pytorch'),
    'pytorch_tabular_tabnet': ('pytorch_tabular_tabnet', 'pytorch'),

    # 'elastic_net': ('elastic_net', 'stand_alone'),
    # 'xgboost': ('xgboost', 'stand_alone'),
    # 'catboost': ('catboost', 'stand_alone'),
    # 'lightgbm': ('lightgbm', 'stand_alone'),
    # 'widedeep_tab_mlp': ('widedeep_tab_mlp', 'pytorch'),
    # 'widedeep_tab_net': ('widedeep_tab_net', 'pytorch'),
    # 'widedeep_saint': ('widedeep_saint', 'pytorch'),
    # 'widedeep_ft_transformer': ('widedeep_ft_transformer', 'pytorch'),
    # 'pytorch_tabular_autoint': ('pytorch_tabular_autoint', 'pytorch'),
    # 'pytorch_tabular_tabnet': ('pytorch_tabular_tabnet', 'pytorch'),
    # 'pytorch_tabular_node': ('pytorch_tabular_node', 'pytorch'),
    # 'nbm_spam_spam': ('nbm_spam_spam', 'pytorch'),
    # 'nbm_spam_nam': ('nbm_spam_nam', 'pytorch'),
    # 'nbm_spam_nbm': ('nbm_spam_nbm', 'pytorch'),
    # 'danet': ('danet', 'pytorch'),
    # 'nam': ('nam', 'pytorch'),
    # 'arm_net_models_armnet': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_armnet_1h': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_afn': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_gc_arm': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_sa_glu': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_dcn': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_cin': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_gcn': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_gat': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_wd': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_ipnn': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_kpnn': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_nfm': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_dcn_plus': ('arm_net_models', 'pytorch'),
    # 'arm_net_models_xdfm': ('arm_net_models', 'pytorch'),
}

for model_name, (model_type, model_framework) in model_dict.items():

    args = f"--multirun " \
           f"hparams_search=immuno/regression/{model_name} " \
           f"experiment=immuno/regression/trn_val " \
           f"model_framework={model_framework} "\
           f"model_type={model_type} " \
           f"logger=none " \
           f"trainer.progress_bar_refresh_rate={progress_bar_refresh_rate} " \
           f"max_epochs={max_epochs} " \
           f"patience={patience} " \
           f"print_config=False " \
           f"base_dir={base_dir} "

    os.system(f"sbatch run_{segment}.sh \"{args}\"")
