import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)
from sa.logging import log_hyperparameters
from pytorch_lightning.loggers import LightningLoggerBase
import pandas as pd
from sa.multiclass.metrics import get_metrics_dict
from src.utils import utils
import xgboost as xgb
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.layout import add_layout
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from typing import List
import wandb


log = utils.get_logger(__name__)

def train_xgboost(config: DictConfig):

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    config.logger.wandb["project"] = config.project_name
    # Init lightning loggers
    loggers: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))

    num_top_features = 20

    log.info("Logging hyperparameters!")
    log_hyperparameters(loggers, config)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()
    feature_names = datamodule.betas.columns.to_list()
    data = pd.merge(datamodule.pheno.loc[:, datamodule.outcome], datamodule.betas, left_index=True, right_index=True)
    train_data = data.iloc[datamodule.ids_train]
    val_data = data.iloc[datamodule.ids_val]
    test_data = data.iloc[datamodule.ids_test]
    X_train = train_data.loc[:, datamodule.betas.columns.values].values
    y_train = train_data.loc[:, datamodule.outcome].values
    X_val = val_data.loc[:, datamodule.betas.columns.values].values
    y_val = val_data.loc[:, datamodule.outcome].values
    X_test = test_data.loc[:, datamodule.betas.columns.values].values
    y_test = test_data.loc[:, datamodule.outcome].values

    dmat_train = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
    dmat_val = xgb.DMatrix(X_val, y_val, feature_names=feature_names)
    dmat_test = xgb.DMatrix(X_test, y_test, feature_names=feature_names)

    class_names = list(datamodule.statuses.keys())

    model_params = {
        'num_class': config.model.output_dim,
        'booster': config.model.booster,
        'eta': config.model.learning_rate,
        'max_depth': config.model.max_depth,
        'gamma': config.model.gamma,
        'sampling_method': config.model.sampling_method,
        'subsample': config.model.subsample,
        'objective': config.model.objective,
        'verbosity': config.model.verbosity,
    }

    num_boost_round = config.trainer.max_epochs
    early_stopping_rounds = config.trainer.patience
    bst = xgb.train(
        params=model_params,
        dtrain=dmat_train,
        evals=[(dmat_train, "train"), (dmat_val, "val"), (dmat_test, "test")],
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds
    )
    bst.save_model(f"epoch_{bst.best_iteration}.model")

    fi = bst.get_score(importance_type='weight')
    feature_importances = pd.DataFrame.from_dict({'feature': list(fi.keys()), 'importance': list(fi.values())})
    feature_importances.sort_values(['importance'], ascending=[False], inplace=True)
    fig = go.Figure()
    ys = feature_importances['feature'][0:num_top_features][::-1]
    xs = feature_importances['importance'][0:num_top_features][::-1]
    add_bar_trace(fig, x=xs, y=ys, text=xs, orientation='h')
    add_layout(fig, f"Feature importance", f"", "")
    fig.update_yaxes(tickfont_size=10)
    fig.update_xaxes(showticklabels=True)
    save_figure(fig, f"feature_importances")
    feature_importances.set_index('feature', inplace=True)
    feature_importances.to_excel("feature_importances.xlsx", index=True)

    y_train_pred_probs = bst.predict(dmat_train)
    y_train_pred = np.argmax(y_train_pred_probs, 1)
    y_val_pred_probs = bst.predict(dmat_val)
    y_val_pred = np.argmax(y_val_pred_probs, 1)
    y_test_pred_probs = bst.predict(dmat_test)
    y_test_pred = np.argmax(y_test_pred_probs, 1)

    metrics_classes_dict = get_metrics_dict(config.model.output_dim, object)
    metrics_summary = {
        'accuracy_macro': 'max',
        'accuracy_weighted': 'max',
        'f1_macro': 'max',
        'cohen_kappa': 'max',
        'matthews_corrcoef': 'max',
        'f1_weighted': 'max',
        'auroc_weighted': 'max',
        'auroc_macro': 'max',
    }
    metrics = [metrics_classes_dict[m]() for m in metrics_summary]
    parts = ['val', 'test', 'train']

    for p in parts:
        for m, sum in metrics_summary.items():
            wandb.define_metric(f"{p}/{m}", summary=sum)

    metrics_dict = {'metric': [m._name for m in metrics]}
    for p in parts:
        if p == 'train':
            y_real = y_train
            y_pred = y_train_pred
            y_pred_probs = y_train_pred_probs
        elif p == 'val':
            y_real = y_val
            y_pred = y_val_pred
            y_pred_probs = y_val_pred_probs
        else:
            y_real = y_test
            y_pred = y_test_pred
            y_pred_probs = y_test_pred_probs
        metrics_dict[p] = []
        log_dict = {}
        for m in metrics:
            if m._name in ['auroc_weighted', 'auroc_macro']:
                m_val = m(y_real, y_pred_probs)
            else:
                m_val = m(y_real, y_pred)
            metrics_dict[p].append(m_val)
            log_dict[f"{p}/{m._name}"] = m_val
        wandb.log(log_dict)

        conf_mtx = confusion_matrix(y_real, y_pred)

        fig = ff.create_annotated_heatmap(conf_mtx, x=class_names, y=class_names, colorscale='Viridis')
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=0.5,
                                y=-0.1,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=-0.33,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        fig.update_layout(margin=dict(t=50, l=200))
        fig['data'][0]['showscale'] = True
        save_figure(fig, f"confusion_matrix_{p}")
    wandb.finish()

    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    metrics_df.set_index('metric', inplace=True)
    metrics_df.to_excel("metrics.xlsx", index=True)

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_df.at[optimized_metric, 'val']

