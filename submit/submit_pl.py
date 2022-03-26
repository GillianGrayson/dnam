import os

model_pl = 'tabnet'

tabnet_n_d_n_a = [8]
tabnet_n_steps = [3]
tabnet_gamma = [1.3]
tabnet_n_independent = [1]
tabnet_n_shared = [2]
tabnet_optimizer_lr = [0.005]
tabnet_optimizer_weight_decay = [0.0]
tabnet_scheduler_step_size = [50]
tabnet_scheduler_gamma = [0.9]

base_dir = "/common/home/yusipov_i/data/dnam/datasets/meta/GPL13534_Blood/Schizophrenia"
in_dim = 200

if model_pl == 'tabnet':
    args = f"--multirun " \
           f"logger=csv " \
           f"base_dir={base_dir} " \
           f"experiment=dnam/multiclass/trn_val_tst/tabnetpl " \
           f"model._target_=src.models.dnam.tabnet.TabNetModel " \
           f"model.input_dim={in_dim} " \
           f"model.n_d_n_a={','.join(str(x) for x in tabnet_n_d_n_a)} " \
           f"model.n_steps={','.join(str(x) for x in tabnet_n_steps)} " \
           f"model.gamma={','.join(str(x) for x in tabnet_gamma)} " \
           f"model.n_independent={','.join(str(x) for x in tabnet_n_independent)} "\
           f"model.n_shared={','.join(str(x) for x in tabnet_n_shared)} " \
           f"model.optimizer_lr={','.join(str(x) for x in tabnet_optimizer_lr)} " \
           f"model.optimizer_weight_decay={','.join(str(x) for x in tabnet_optimizer_weight_decay)} " \
           f"model.scheduler_step_size={','.join(str(x) for x in tabnet_scheduler_step_size)} " \
           f"model.scheduler_gamma={','.join(str(x) for x in tabnet_scheduler_gamma)} " \
           f"trainer.gpus=0 "

os.system(f"sbatch run_multiclass_trn_val_tst_pl.sh \"{args}\"")
