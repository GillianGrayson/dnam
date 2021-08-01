import plotly.graph_objects as go


def add_bar_trace(fig, x, y, name=""):
    showlegend = False if name == "" else True
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name=name,
            textposition='auto',
            showlegend=showlegend
        )
    )
