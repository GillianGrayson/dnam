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


disease = "Parkinson"
colors = ["red", "blue"]
datasets = {"GSE72774": "accuracy_weighted_val"}
path = f"E:/YandexDisk/Work/pydnameth/draft/03_somewhere/Figure4/dim_red/{disease}"

fig = go.Figure()
for ds in datasets:
    df = pd.read_excel(f"{path}/{ds}/metrics.xlsx", index_col='n_feat')
    add_scatter_trace(fig, df.index.values, df.loc[:, datasets[ds]], ds, mode='lines+markers')
    add_layout(fig, f"Number of features in model", f"Accuracy", "")
    fig.update_layout({'colorway': colors})
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
    save_figure(fig, f"{path}/accuracy")