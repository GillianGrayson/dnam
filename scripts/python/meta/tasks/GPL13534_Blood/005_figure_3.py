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


disease = "Schizophrenia"
dataset = "GSE152027"
data_type = "non_harmonized"
metric = "accuracy_weighted"
path = f"E:/YandexDisk/Work/pydnameth/draft/03_somewhere/Figure3/baseline"

df = pd.read_excel(f"{path}/{disease}/{dataset}/{data_type}/bar.xlsx")

models = df.loc[:, "model"].values
metrics = df.loc[:, metric].values
colors = px.colors.qualitative.Set1[0:len(models)]
order = metrics.argsort()

fig = go.Figure()
for index in order:
    fig.add_trace(
        go.Bar(
            x=[metrics[index]],
            y=[models[index]],
            name=models[index],
            text=metrics[index],
            textfont=dict(size=20),
            textposition='auto',
            orientation='h',
            showlegend=False,
            marker=dict(
                color=colors[index],
                line=dict(color='black', width=2)
            )
        )
    )
add_layout(fig, f"Accuracy", f"", "")
fig.update_yaxes(tickfont_size=20)
fig.update_xaxes(showticklabels=True)
fig.update_layout(legend={'itemsizing': 'constant'})
fig.update_layout(margin=go.layout.Margin(l=180, r=20, b=80, t=25, pad=0))
save_figure(fig, f"{path}/{disease}/{dataset}/{data_type}/bar")