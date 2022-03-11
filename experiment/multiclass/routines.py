import pandas as pd
from experiment.multiclass.metrics import get_metrics_dict
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import wandb
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout


def eval_classification(config, part, class_names, y_real, y_pred, y_pred_probs, loggers, probs=True):
    metrics_classes_dict = get_metrics_dict(config.out_dim, object)
    metrics_summary = {
        'accuracy_macro': 'max',
        'accuracy_micro': 'max',
        'accuracy_weighted': 'max',
        'f1_macro': 'max',
        'cohen_kappa': 'max',
        'matthews_corrcoef': 'max',
        'f1_weighted': 'max'
    }
    if probs:
        metrics_summary['auroc_weighted'] = 'max'
        metrics_summary['auroc_macro'] = 'max'

    metrics = [metrics_classes_dict[m]() for m in metrics_summary]

    for m, sum in metrics_summary.items():
        wandb.define_metric(f"{part}/{m}", summary=sum)

    metrics_dict = {'metric': [m._name for m in metrics]}
    metrics_dict[part] = []
    log_dict = {}
    for m in metrics:
        if m._name in ['auroc_weighted', 'auroc_macro']:
            m_val = m(y_real, y_pred_probs)
        else:
            m_val = m(y_real, y_pred)
        metrics_dict[part].append(m_val)
        log_dict[f"{part}/{m._name}"] = m_val
    #wandb.log(log_dict)
    for logger in loggers:
        logger.log_metrics(log_dict)

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
    save_figure(fig, f"confusion_matrix_{part}")

    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    metrics_df.set_index('metric', inplace=True)
    metrics_df.to_excel(f"metrics_{part}.xlsx", index=True)

    return metrics_df


def eval_loss(loss_info, loggers):
    for epoch_id, epoch in enumerate(loss_info['epoch']):
        log_dict = {
            'epoch': loss_info['epoch'][epoch_id],
            'train/loss': loss_info['train/loss'][epoch_id],
            'val/loss': loss_info['val/loss'][epoch_id]
        }
        #wandb.log(log_dict)
        for logger in loggers:
            logger.log_metrics(log_dict)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=loss_info['epoch'],
            y=loss_info['train/loss'],
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
    fig.update_layout(yaxis_range=[0, max(loss_info['train/loss'] + loss_info['val/loss']) + 0.1])
    save_figure(fig, f"loss")
