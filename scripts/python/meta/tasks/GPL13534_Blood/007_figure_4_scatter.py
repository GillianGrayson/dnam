import pandas as pd
import plotly.graph_objects as go
from scripts.python.routines.plot.save import save_figure
from scripts.python.routines.plot.layout import add_layout
import plotly.io as pio
pio.kaleido.scope.mathjax = None


disease = "Parkinson"
color = "green"
path = f"E:/YandexDisk/Work/pydnameth/draft/03_somewhere/Figure4/dim_red/{disease}"

n_feat = 10

df = pd.read_excel(f"{path}/feature_importances.xlsx")
df['importance'] = df['importance'] / df['importance'].sum()

ys = df.loc[range(n_feat), 'feature'].values[::-1]
xs = df.loc[range(n_feat), 'importance'].values[::-1]

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=xs,
        y=list(range(len(ys))),
        orientation='h',
        marker=dict(color=color, opacity=0.9)
    )
)
add_layout(fig, "Feature importance", "", f"")
fig.update_layout({'colorway': [color]})
fig.update_layout(legend_font_size=20)
fig.update_layout(showlegend=False)
fig.update_layout(
    yaxis = dict(
        tickmode = 'array',
        tickvals = list(range(len(xs))),
        ticktext = ys
    )
)
fig.update_yaxes(autorange=False)
fig.update_layout(yaxis_range=[-0.5, len(xs)-0.5])
fig.update_yaxes(tickfont_size=24)
fig.update_xaxes(tickfont_size=24)
fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    margin=go.layout.Margin(
        l=175,
        r=20,
        b=100,
        t=40,
        pad=0
    )
)
save_figure(fig, f"{path}/feature_importances")
