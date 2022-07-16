import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.bar import add_bar_trace
from scripts.python.routines.plot.layout import add_layout
from scripts.python.routines.manifest import get_manifest
from scripts.python.pheno.datasets.filter import filter_pheno, get_passed_fields
from scripts.python.pheno.datasets.features import get_column_name, get_status_dict_default
from scripts.python.preprocessing.serialization.routines.pheno_betas_checking import get_pheno_betas_with_common_subjects
from scripts.python.routines.betas import betas_drop_na
from scripts.python.routines.mvals import logit2
from scripts.python.meta.tasks.GPL13534_Blood.routines import perform_test_for_controls
from tqdm import tqdm
import pathlib
import plotly.express as px
import plotly.io as pio
pio.kaleido.scope.mathjax = None
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.scatter import add_scatter_trace
from scripts.python.routines.plot.layout import add_layout


disease = "Schizophrenia" # "Parkinson" "Schizophrenia"
colors = ["red", "blue"]
datasets = {"GSE152027": "accuracy_weighted_test", "GSE116379": "accuracy_weighted_test"} # {"GSE152027": "accuracy_weighted_test", "GSE116379": "accuracy_weighted_test"} {"GSE72774": "accuracy_weighted_val"}
path = f"E:/YandexDisk/Work/pydnameth/draft/03_somewhere/revision/Figure5/dim_red/{disease}"

x_min = -20
x_max = 1050

y_min = 0.61
y_max = 0.85

fig = go.Figure()
for ds_id, ds in enumerate(datasets):
    df = pd.read_excel(f"{path}/{ds}/metrics.xlsx", index_col='n_feat')
    fig.add_trace(
        go.Scatter(
            x=df.index.values,
            y=df.loc[:, datasets[ds]].values,
            showlegend=True,
            name=ds,
            mode="lines+markers",
            marker=dict(
                size=8,
                opacity=0.7,
                line=dict(
                    width=0.0
                )
            )
        )
    )
add_layout(fig, f"Number of features in model", f"Accuracy", "")
fig.update_layout({'colorway': colors})

for ds_id, ds in enumerate(datasets):
    df = pd.read_excel(f"{path}/{ds}/metrics.xlsx", index_col='n_feat')
    if ds_id == 0:
        index_opt = np.argmax(df.loc[:, datasets[ds]].values)
        x_opt = df.index.values[index_opt]
        y_opt = df.loc[:, datasets[ds]].values[index_opt]

        fig.add_trace(
            go.Scatter(
                x=[x_min, x_opt],
                y=[y_opt, y_opt],
                showlegend=False,
                mode='lines',
                line=dict(color='black', width=2, dash='dot')
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[x_opt, x_opt],
                y=[0, y_opt],
                showlegend=False,
                mode='lines',
                line=dict(color='black', width=2, dash='dot')
            )
        )

fig.update_layout(legend_font_size=20)
fig.update_layout(legend={'itemsizing': 'constant'})
fig.update_layout(
    margin=go.layout.Margin(
        l=130,
        r=20,
        b=80,
        t=80,
        pad=0
    )
)
fig.update_yaxes(autorange=False)
fig.update_layout(yaxis_range=[y_min, y_max])
fig.update_xaxes(autorange=False)
fig.update_layout(xaxis_range=[x_min, x_max])
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
save_figure(fig, f"{path}/accuracy")