import pandas as pd
from experiment.multiclass.metrics import get_metrics_dict
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import wandb
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout


def eval_classification_sa(config, part, class_names, y_real, y_pred, y_pred_probs, loggers, probs=True):
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

    if 'wandb' in config.logger:
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
    for logger in loggers:
        logger.log_metrics(log_dict)

    plot_confusion_matrix(y_real, y_pred, class_names, part)

    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    metrics_df.set_index('metric', inplace=True)
    metrics_df.to_excel(f"metrics_{part}.xlsx", index=True)

    return metrics_df


def plot_confusion_matrix(y_real, y_pred, class_names, part):
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
