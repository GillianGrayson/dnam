import pandas as pd
from experiment.multiclass.metrics import get_metrics_dict
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import wandb
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout


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
