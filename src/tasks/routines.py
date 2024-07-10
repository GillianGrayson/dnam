import pandas as pd
from src.tasks.metrics import get_cls_pred_metrics, get_cls_prob_metrics, get_reg_metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import wandb
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
import torch


def save_feature_importance(df, num_features, config='none'):
    if config != 'none':
        if df is not None:
            df.sort_values(['importance'], ascending=[False], inplace=True)
            df['importance'] = df['importance'] / df['importance'].sum()
            df_fig = df.iloc[0:num_features, :]
            plt.figure(figsize=(8, 0.3 * df_fig.shape[0]))
            sns.set_theme(style='whitegrid', font_scale=1)
            bar = sns.barplot(
                data=df_fig,
                y='feature_label',
                x='importance',
                edgecolor='black',
                orient='h',
                dodge=True
            )
            bar.set_xlabel("Importance")
            bar.set_ylabel("")
            plt.savefig(f"feature_importance.png", bbox_inches='tight', dpi=400)
            plt.savefig(f"feature_importance.pdf", bbox_inches='tight')
            plt.close()
            df.set_index('feature', inplace=True)
            df.to_excel("feature_importance.xlsx", index=True)


def eval_classification(config, class_names, y_real, y_pred, y_pred_prob, loggers, part, is_log=True, is_save=True, file_suffix=''):
    metrics_pred = get_cls_pred_metrics(config.out_dim)
    metrics_prob = get_cls_prob_metrics(config.out_dim)

    if is_log:
        if 'wandb' in config.logger:
            for m in metrics_pred:
                wandb.define_metric(f"{part}/{m}", summary=metrics_pred[m][1])
            for m in metrics_prob:
                wandb.define_metric(f"{part}/{m}", summary=metrics_prob[m][1])

    metrics_df = pd.DataFrame(index=[m for m in metrics_pred] + [m for m in metrics_prob], columns=[part])
    metrics_df.index.name = 'metric'
    log_dict = {}
    for m in metrics_pred:
        y_real_torch = torch.from_numpy(y_real)
        y_pred_torch = torch.from_numpy(y_pred)
        m_val = float(metrics_pred[m][0](y_pred_torch, y_real_torch).numpy())
        metrics_pred[m][0].reset()
        metrics_df.at[m, part] = m_val
        log_dict[f"{part}/{m}"] = m_val
    for m in metrics_prob:
        y_real_torch = torch.from_numpy(y_real)
        y_pred_prob_torch = torch.from_numpy(y_pred_prob)
        m_val = 0
        try:
            m_val = float(metrics_prob[m][0](y_pred_prob_torch, y_real_torch).numpy())
        except ValueError:
            pass
        metrics_prob[m][0].reset()
        metrics_df.at[m, part] = m_val
        log_dict[f"{part}/{m}"] = m_val

    if loggers is not None:
        for logger in loggers:
            if is_log:
                logger.log_metrics(log_dict)

    if is_save:
        plot_confusion_matrix(y_real, y_pred, class_names, part, suffix=file_suffix)

    return metrics_df


def eval_regression(config, y_real, y_pred, loggers, part, is_log=True, is_save=True, file_suffix=''):
    metrics = get_reg_metrics()

    if is_log:
        if 'wandb' in config.logger:
            for m in metrics:
                wandb.define_metric(f"{part}/{m}", summary=metrics[m][1])

    metrics_df = pd.DataFrame(index=[m for m in metrics], columns=[part])
    metrics_df.index.name = 'metric'
    log_dict = {}
    for m in metrics:
        y_real_torch = torch.from_numpy(y_real)
        y_pred_torch = torch.from_numpy(y_pred)
        m_val = float(metrics[m][0](y_pred_torch, y_real_torch).numpy())
        metrics[m][0].reset()
        metrics_df.at[m, part] = m_val
        log_dict[f"{part}/{m}"] = m_val

    if loggers is not None:
        for logger in loggers:
            if is_log:
                logger.log_metrics(log_dict)

    if is_save:
        metrics_df.to_excel(f"metrics_{part}{file_suffix}.xlsx", index=True)

    return metrics_df


def plot_confusion_matrix(y_real, y_pred, class_names, part, suffix=''):
    cm = confusion_matrix(y_real, y_pred)
    if len(cm) > 1:
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=(2*len(class_names), 2*len(class_names)))
        sns.heatmap(cm, annot=annot, fmt='', ax=ax)
        plt.savefig(f"confusion_matrix_{part}{suffix}.png", bbox_inches='tight')
        plt.savefig(f"confusion_matrix_{part}{suffix}.pdf", bbox_inches='tight')
        plt.close()


def eval_loss(loss_info, loggers, is_log=True, is_save=True, file_suffix=''):
    for epoch_id, epoch in enumerate(loss_info['epoch']):
        log_dict = {
            'epoch': loss_info['epoch'][epoch_id],
            'trn/loss': loss_info['trn/loss'][epoch_id],
            'val/loss': loss_info['val/loss'][epoch_id]
        }
        if loggers is not None:
            for logger in loggers:
                if is_log:
                    logger.log_metrics(log_dict)

    if is_save:
        loss_df = pd.DataFrame(loss_info)
        loss_df.set_index('epoch', inplace=True)
        loss_df.to_excel(f"loss{file_suffix}.xlsx", index=True)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=loss_info['epoch'],
                y=loss_info['trn/loss'],
                showlegend=True,
                name="Train",
                mode="lines",
                marker=dict(
                    size=8,
                    opacity=0.7,
                    line=dict(
                        width=1
                    )
                )
            )
        )
        fig.add_trace(
            go.Scatter(
                x=loss_info['epoch'],
                y=loss_info['val/loss'],
                showlegend=True,
                name="Val",
                mode="lines",
                marker=dict(
                    size=8,
                    opacity=0.7,
                    line=dict(
                        width=1
                    )
                )
            )
        )
        add_layout(fig, "Epoch", 'Error', "")
        fig.update_layout({'colorway': ['blue', 'red']})
        fig.update_layout(legend_font_size=20)
        fig.update_layout(
            margin=go.layout.Margin(
                l=90,
                r=20,
                b=75,
                t=45,
                pad=0
            )
        )
        fig.update_yaxes(autorange=False)
        fig.update_layout(yaxis_range=[0, max(loss_info['trn/loss'] + loss_info['val/loss']) + 0.1])
        save_figure(fig, f"loss{file_suffix}")
