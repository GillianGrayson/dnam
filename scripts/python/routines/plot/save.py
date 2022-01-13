import plotly
import time

def save_figure(fig, fn, width=800, height=600, scale=2):
    fig.write_image(f"{fn}.png")
    fig.write_image(f"{fn}.pdf", format="pdf")
    #time.sleep(2)
    fig.write_image(f"{fn}.pdf", format="pdf")
    #plotly.io.write_image(fig, f"{fn}.png", width=width, height=height, scale=scale)
    #plotly.io.write_image(fig, f"{fn}.pdf", width=width, height=height, scale=scale)
