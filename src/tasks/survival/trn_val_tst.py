from typing import List, Optional
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    seed_everything,
)
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
import seaborn as sns
from src.datamodules.cross_validation import RepeatedStratifiedKFoldCVSplitter
from src.datamodules.tabular import TabularDataModule
import numpy as np
from src.utils import utils
import pandas as pd
from tqdm import tqdm
import pathlib
import pickle
import matplotlib.pyplot as plt


log = utils.get_logger(__name__)


def trn_val_tst_survival(config: DictConfig) -> Optional[float]:

    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: TabularDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.perform_split()
    features = datamodule.get_features()
    num_features = len(features['all'])
    config.in_dim = num_features
    event_names = datamodule.get_class_names()
    event = datamodule.target
    event_label = datamodule.target_label
    duration = datamodule.duration
    duration_label = datamodule.duration_label

    df = datamodule.get_data()

    ids_tst = datamodule.ids_tst

    cv_splitter = RepeatedStratifiedKFoldCVSplitter(
        datamodule=datamodule,
        is_split=config.cv_is_split,
        n_splits=config.cv_n_splits,
        n_repeats=config.cv_n_repeats,
        random_state=config.seed
    )

    best = {}
    if config.direction == "min":
        best["optimized_metric"] = np.Inf
    elif config.direction == "max":
        best["optimized_metric"] = 0.0

    metrics_cv = pd.DataFrame()
    feat_imps_cv = pd.DataFrame(index=features['all'])

    try:
        for fold_id, (ids_trn, ids_val) in tqdm(enumerate(cv_splitter.split())):
            datamodule.ids_trn = ids_trn
            datamodule.ids_val = ids_val
            datamodule.refresh_datasets()

            dfs = {
                'trn': df.loc[df.index[ids_trn], list(features['all']) + [event, duration]],
                'val': df.loc[df.index[ids_val], list(features['all']) + [event, duration]]
            }
            df.loc[df.index[ids_trn], f"fold_{fold_id:04d}"] = "trn"
            df.loc[df.index[ids_val], f"fold_{fold_id:04d}"] = "val"
            for tst_set_name in ids_tst:
                dfs[tst_set_name] = df.loc[df.index[ids_tst[tst_set_name]], list(features['all']) + [event, duration]]
                if tst_set_name != 'tst_all':
                    df.loc[df.index[ids_tst[tst_set_name]], f"fold_{fold_id:04d}"] = tst_set_name

            if config.model.name == "coxnet":
                model = CoxnetSurvivalAnalysis(
                    l1_ratio=config.model.l1_ratio,
                    alphas=[config.model.alpha],
                    tol=config.model.tol,
                    max_iter=config.model.max_iter,
                    fit_baseline_model=True
                )
                X_trn = dfs['trn'].loc[:, features['all']]
                y_trn = np.zeros(dfs['trn'].shape[0], dtype={'names': ('event', 'duration'), 'formats': (np.bool, np.float32)})
                y_trn['event'] = dfs['trn'].loc[:, event].values
                y_trn['duration'] = dfs['trn'].loc[:, duration].values
                model.fit(X_trn, y_trn)

                feat_imps_cv.loc[features['all'], fold_id] = model.coef_

                for df_part in dfs:
                    X = dfs[df_part].loc[:, features['all']]
                    y = np.zeros(dfs[df_part].shape[0], dtype={'names': ('event', 'duration'), 'formats': (np.bool, np.float32)})
                    y['event'] = dfs[df_part].loc[:, event].values
                    y['duration'] = dfs[df_part].loc[:, duration].values
                    metrics_cv.at[fold_id, f"{df_part}_ci"] = model.score(X, y)

            is_renew = False
            if config.direction == "min":
                if metrics_cv.at[fold_id, f"{config.optimized_part}_{config.optimized_metric}"] < best["optimized_metric"]:
                    is_renew = True
            elif config.direction == "max":
                if metrics_cv.at[fold_id, f"{config.optimized_part}_{config.optimized_metric}"] > best["optimized_metric"]:
                    is_renew = True

            if is_renew:
                best["optimized_metric"] = metrics_cv.at[fold_id, f"{config.optimized_part}_{config.optimized_metric}"]
                best["model"] = model
                best['fold_id'] = fold_id
                best['ids_trn'] = ids_trn
                best['ids_val'] = ids_val

        metrics_cv.to_excel(f"metrics_cv.xlsx", index_label='fold')
        feat_imps_cv.to_excel(f"feat_imps_cv.xlsx", index_label='feature')
        cv_ids_cols = [f"fold_{fold_id:04d}" for fold_id in metrics_cv.index.values]
        if datamodule.split_top_feat:
            cv_ids_cols.append(datamodule.split_top_feat)
        cv_ids = df.loc[:, cv_ids_cols]
        cv_ids.to_excel(f"cv_ids.xlsx", index=True)

        datamodule.ids_trn = best['ids_trn']
        datamodule.ids_val = best['ids_val']

        datamodule.plot_split(f"_best_{best['fold_id']:04d}")

        pickle.dump(best["model"], open(f"model_{best['fold_id']:04d}.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        metrics_names = ['ci']
        parts = ['trn', 'val'] + list(ids_tst.keys())
        metrics = pd.DataFrame(index=metrics_names, columns=parts)
        for m in metrics_names:
            for p in parts:
                metrics.at[m, p] = metrics_cv.at[best['fold_id'], f'{p}_{m}']
                metrics.at[m, f'{p}_mean'] = metrics_cv.loc[:, f'{p}_{m}'].mean()
            for tst_set_name in ids_tst:
                metrics.at[m, f'val_{tst_set_name}_mean'] = 0.5 * (metrics.at[m, 'val'] + metrics.at[m, tst_set_name])
        metrics.to_excel(f"cv_ids.xlsx", index_label='metric')

        X_all = df.loc[:, features['all']]
        if config.model.name == "coxnet":
            event_times = best["model"].event_times_
            surv_func = best["model"].predict_survival_function(X_all, alpha=config.model.alpha, return_array=True)
            df_surv_func = pd.DataFrame(index=X_all.index.values, columns=event_times, data=surv_func)

        pathlib.Path(f"surv_func").mkdir(parents=True, exist_ok=True)
        for cat_feat in features['cat']:
            list_surv_func = []
            for time in df_surv_func.columns.values:
                dict_time = {
                    'Sample': df_surv_func.index.values,
                    't': [time] * df_surv_func.shape[0],
                    'S(t)': df_surv_func.loc[df_surv_func.index.values, time].values,
                    f"{features['labels'][cat_feat]}": X_all.loc[df_surv_func.index.values, cat_feat]
                }
                list_surv_func.append(pd.DataFrame(dict_time))
            df_fig = pd.concat(list_surv_func, ignore_index=True)
            df_fig.replace({features['labels'][cat_feat]: datamodule.dict_cat_replace[cat_feat]}, inplace=True)
            fig = plt.figure()
            sns.set_theme(style='whitegrid', font_scale=1)
            palette = datamodule.dict_colors[cat_feat]
            sns.lineplot(
                data=df_fig,
                x='t',
                y="S(t)",
                hue=f"{features['labels'][cat_feat]}",
                palette=palette,
                hue_order=list(palette.keys())
            )
            plt.savefig(f"surv_func/{cat_feat}.png", bbox_inches='tight', dpi=400)
            plt.savefig(f"surv_func/{cat_feat}.pdf", bbox_inches='tight')
            plt.close(fig)
        for con_feat in features['con']:
            palette = {"<Q1": 'lawngreen', "Q1-Q2": 'gold', "Q2-Q3": 'orangered', ">Q3": 'firebrick'}
            q_labels = pd.qcut(X_all.loc[df_surv_func.index.values, con_feat], q=4, labels=["<Q1", "Q1-Q2", "Q2-Q3", ">Q3"])
            list_surv_func = []
            for time in df_surv_func.columns.values:
                dict_time = {
                    'Sample': df_surv_func.index.values,
                    't': [time] * df_surv_func.shape[0],
                    'S(t)': df_surv_func.loc[df_surv_func.index.values, time].values,
                    f"{features['labels'][con_feat]}": q_labels.values
                }
                list_surv_func.append(pd.DataFrame(dict_time))
            df_fig = pd.concat(list_surv_func, ignore_index=True)
            fig = plt.figure()
            sns.set_theme(style='whitegrid', font_scale=1)
            sns.lineplot(
                data=df_fig,
                x='t',
                y="S(t)",
                hue=f"{features['labels'][con_feat]}",
                palette=palette,
                hue_order=list(palette.keys())
            )
            plt.savefig(f"surv_func/{con_feat}.png", bbox_inches='tight', dpi=400)
            plt.savefig(f"surv_func/{con_feat}.pdf", bbox_inches='tight')
            plt.close(fig)

        # Return metric score for hyperparameter optimization
        if config.optimized_metric:
            if config.optimized_mean == "":
                return metrics.at[config.optimized_metric, config.optimized_part]
            else:
                return metrics.at[config.optimized_metric, f"{config.optimized_part}_{config.optimized_mean}"]

    except ArithmeticError:
        log.error(f"Numerical error during {config.model.name}")
        return  best["optimized_metric"]
