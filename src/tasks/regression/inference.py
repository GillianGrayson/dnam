import numpy as np
import torch
import lightgbm as lgb
import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from src.datamodules.tabular import TabularDataModule
from src.utils import utils
import statsmodels.formula.api as smf
import xgboost as xgb
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
from src.tasks.routines import eval_regression
from catboost import CatBoost
from scripts.python.routines.plot.scatter import add_scatter_trace
import pandas as pd
from src.tasks.regression.shap import explain_shap
from src.tasks.regression.lime import explain_lime
from scipy.stats import mannwhitneyu
from scripts.python.routines.plot.p_value import add_p_value_annotation


log = utils.get_logger(__name__)

def inference(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

    if 'wandb' in config.logger:
        config.logger.wandb["project"] = config.project_name

    # Init Lightning datamodule for test
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: TabularDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.perform_split()
    feature_names = datamodule.get_feature_names()
    num_features = len(feature_names['all'])
    config.in_dim = num_features
    target_name = datamodule.get_target()
    df = datamodule.get_data()

    df_parts = pd.read_excel(config.path_parts, index_col="index").iloc[:, [0]]
    df_parts.rename(columns={df_parts.columns.values[0]: "part"}, inplace=True)
    indexes_trn = df_parts.loc[df_parts["part"] == "trn", :].index.values
    indexes_val = df_parts.loc[df_parts["part"] == "val", :].index.values
    indexes_tst = df_parts.loc[df_parts["part"] == "tst", :].index.values

    is_tst = True if len(indexes_tst) > 0 else False
    X_trn = df.loc[indexes_trn, feature_names['all']].values
    y_trn = df.loc[indexes_trn, target_name].values
    df.loc[indexes_trn, "part"] = "trn"
    X_val = df.loc[indexes_val, feature_names['all']].values
    y_val = df.loc[indexes_val, target_name].values
    df.loc[indexes_val, "part"] = "val"
    if is_tst:
        X_tst = df.loc[indexes_tst, feature_names['all']].values
        y_tst = df.loc[indexes_tst, target_name].values
        df.loc[indexes_tst, "part"] = "tst"

    if config.model_framework == "pytorch":
        config.model = config[config.model_type]

        widedeep = datamodule.get_widedeep()
        embedding_dims = [(x[1], x[2]) for x in widedeep['cat_embed_input']] if widedeep['cat_embed_input'] else []
        if config.model_type.startswith('widedeep'):
            config.model.column_idx = widedeep['column_idx']
            config.model.cat_embed_input = widedeep['cat_embed_input']
            config.model.continuous_cols = widedeep['continuous_cols']
        elif config.model_type.startswith('pytorch_tabular'):
            config.model.continuous_cols = feature_names['con']
            config.model.categorical_cols = feature_names['cat']
            config.model.embedding_dims = embedding_dims
        elif config.model_type == 'nam':
            num_unique_vals = [len(np.unique(X_trn[:, i])) for i in range(X_trn.shape[1])]
            num_units = [min(config.model.num_basis_functions, i * config.model.units_multiplier) for i in
                         num_unique_vals]
            config.model.num_units = num_units
        log.info(f"Instantiating model <{config.model._target_}>")
        model = hydra.utils.instantiate(config.model)

        model = type(model).load_from_checkpoint(checkpoint_path=f"{config.path_ckpt}")
        model.eval()
        model.freeze()

        y_trn_pred = model(torch.from_numpy(X_trn)).cpu().detach().numpy().ravel()
        y_val_pred = model(torch.from_numpy(X_val)).cpu().detach().numpy().ravel()
        if is_tst:
            y_tst_pred = model(torch.from_numpy(X_tst)).cpu().detach().numpy().ravel()

        def predict_func(X):
            batch = {
                'all': torch.from_numpy(np.float32(X[:, feature_names['all_ids']])),
                'continuous': torch.from_numpy(np.float32(X[:, feature_names['con_ids']])),
                'categorical': torch.from_numpy(np.float32(X[:, feature_names['cat_ids']])),
            }
            tmp = model(batch)
            return tmp.cpu().detach().numpy()

    elif config.model_framework == "stand_alone":
        if config.model_type == "xgboost":
            model = xgb.Booster()
            model.load_model(config.path_ckpt)

            dmat_trn = xgb.DMatrix(X_trn, y_trn, feature_names=feature_names['all'])
            y_trn_pred = model.predict(dmat_trn)
            dmat_val = xgb.DMatrix(X_val, y_val, feature_names=feature_names['all'])
            y_val_pred = model.predict(dmat_val)
            if is_tst:
                dmat_tst = xgb.DMatrix(X_tst, y_tst, feature_names=feature_names['all'])
                y_tst_pred = model.predict(dmat_tst)

            def predict_func(X):
                X = xgb.DMatrix(X, feature_names=feature_names['all'])
                y = model.predict(X)
                return y

        elif config.model_type == "catboost":
            model = CatBoost()
            model.load_model(config.path_ckpt)

            y_trn_pred = model.predict(X_trn).astype('float32')
            y_val_pred = model.predict(X_val).astype('float32')
            if is_tst:
                y_tst_pred = model.predict(X_tst).astype('float32')

            def predict_func(X):
                y = model.predict(X)
                return y
        elif config.model_type == "lightgbm":
            model = lgb.Booster(model_file=config.path_ckpt)

            y_trn_pred = model.predict(X_trn, num_iteration=model.best_iteration).astype('float32')
            y_val_pred = model.predict(X_val, num_iteration=model.best_iteration).astype('float32')
            if is_tst:
                y_tst_pred = model.predict(X_tst, num_iteration=model.best_iteration).astype('float32')

            def predict_func(X):
                y = model.predict(X, num_iteration=model.best_iteration)
                return y
        else:
            raise ValueError(f"Model {config.model_type} is not supported")

    else:
        raise ValueError(f"Unsupported model_framework: {config.model_framework}")

    df.loc[indexes_trn, "Estimation"] = y_trn_pred
    df.loc[indexes_val, "Estimation"] = y_val_pred
    if is_tst:
        df.loc[indexes_tst, "Estimation"] = y_tst_pred

    eval_regression(config, y_trn, y_trn_pred, None, 'trn', is_log=False, is_save=True, file_suffix=f"")
    eval_regression(config, y_val, y_val_pred, None, 'val', is_log=False, is_save=True, file_suffix=f"")
    if is_tst:
        eval_regression(config, y_tst, y_tst_pred, None, 'tst', is_log=False, is_save=True, file_suffix=f"")

    formula = f"Estimation ~ {target_name}"
    model_linear = smf.ols(formula=formula, data=df.loc[indexes_trn, :]).fit()
    df.loc[indexes_trn, "Estimation acceleration"] = df.loc[indexes_trn, "Estimation"].values - model_linear.predict(df.loc[indexes_trn, :])
    df.loc[indexes_val, "Estimation acceleration"] = df.loc[indexes_val, "Estimation"].values - model_linear.predict(df.loc[indexes_val, :])
    if is_tst:
        df.loc[indexes_tst, "Estimation acceleration"] = df.loc[indexes_tst, "Estimation"].values - model_linear.predict(df.loc[indexes_tst, :])
    fig = go.Figure()
    add_scatter_trace(fig, df.loc[indexes_trn, target_name].values, df.loc[indexes_trn, "Estimation"].values, f"Train")
    add_scatter_trace(fig, df.loc[indexes_trn, target_name].values, model_linear.fittedvalues.values, "", "lines")
    add_scatter_trace(fig, df.loc[indexes_val, target_name].values, df.loc[indexes_val, "Estimation"].values, f"Val")
    if is_tst:
        add_scatter_trace(fig, df.loc[indexes_tst, target_name].values, df.loc[indexes_tst, "Estimation"].values, f"Test")
    add_layout(fig, target_name, f"Estimation", f"")
    fig.update_layout({'colorway': ['blue', 'blue', 'red', 'green']})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(margin=go.layout.Margin(l=90, r=20, b=80, t=65, pad=0))
    save_figure(fig, f"scatter")

    dist_num_bins = 15
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=df.loc[indexes_trn, "Estimation acceleration"].values,
            name=f"Train",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor='blue',
            marker=dict(color='blue', line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(df.loc[indexes_trn, "Estimation acceleration"].values) / dist_num_bins,
            opacity=0.8
        )
    )
    fig.add_trace(
        go.Violin(
            y=df.loc[indexes_val, "Estimation acceleration"].values,
            name=f"Val",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            line_color='black',
            fillcolor='red',
            marker=dict(color='red', line=dict(color='black', width=0.3), opacity=0.8),
            points='all',
            bandwidth=np.ptp(df.loc[indexes_val, "Estimation acceleration"].values) / dist_num_bins,
            opacity=0.8
        )
    )
    if is_tst:
        fig.add_trace(
            go.Violin(
                y=df.loc[indexes_tst, "Estimation acceleration"].values,
                name=f"Test",
                box_visible=True,
                meanline_visible=True,
                showlegend=True,
                line_color='black',
                fillcolor='green',
                marker=dict(color='green', line=dict(color='black', width=0.3), opacity=0.8),
                points='all',
                bandwidth=np.ptp(df.loc[indexes_tst, "Estimation acceleration"].values) / 50,
                opacity=0.8
            )
        )
    add_layout(fig, "", "Estimation acceleration", f"")
    fig.update_layout({'colorway': ['red', 'blue', 'green']})
    stat_01, pval_01 = mannwhitneyu(
        df.loc[indexes_trn, "Estimation acceleration"].values,
        df.loc[indexes_val, "Estimation acceleration"].values,
        alternative='two-sided'
    )
    if is_tst:
        stat_02, pval_02 = mannwhitneyu(
            df.loc[indexes_trn, "Estimation acceleration"].values,
            df.loc[indexes_tst, "Estimation acceleration"].values,
            alternative='two-sided'
        )
        stat_12, pval_12 = mannwhitneyu(
            df.loc[indexes_val, "Estimation acceleration"].values,
            df.loc[indexes_tst, "Estimation acceleration"].values,
            alternative='two-sided'
        )
        fig = add_p_value_annotation(fig, {(0, 1): pval_01, (1, 2): pval_12, (0, 2): pval_02})
    else:
        fig = add_p_value_annotation(fig, {(0, 1): pval_01})
    fig.update_layout(title_xref='paper')
    fig.update_layout(legend_font_size=20)
    fig.update_layout(
        margin=go.layout.Margin(
            l=110,
            r=20,
            b=50,
            t=90,
            pad=0
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.25,
            xanchor="center",
            x=0.5
        )
    )
    save_figure(fig, f"violin")

    df['ids'] = np.arange(df.shape[0])
    ids_trn = df.loc[indexes_trn, 'ids'].values
    ids_val = df.loc[indexes_val, 'ids'].values
    ids_tst = df.loc[indexes_tst, 'ids'].values

    expl_data = {
        'model': model,
        'predict_func': predict_func,
        'df': df,
        'feature_names': feature_names['all'],
        'target_name': target_name,
        'ids_all': np.arange(df.shape[0]),
        'ids_trn': ids_trn,
        'ids_val': ids_val,
        'ids_tst': ids_tst
    }
    if config.is_lime == True:
        explain_lime(config, expl_data)
    if config.is_shap == True:
        explain_shap(config, expl_data)

    df.to_excel("df.xlsx", index=True)
