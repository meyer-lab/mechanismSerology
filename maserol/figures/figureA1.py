"""
This is an internal figure which does not necessarily have a place in final
paper.
"""
from tensordata.zohar import data3D as zohar

from maserol.core import prepare_data
from maserol.figures.heatmap import plot_deviation_heatmap
from maserol.preprocess import HIgGs
from maserol.validation import bootstrap

def makeFigure():
    data = prepare_data(zohar(xarray=True, logscale=False))
    opt_kwargs = {
        "metric": "rtot",
        "lrank": True,
        "fitKa": False,
    }
    sample_dist, ag_dist = bootstrap(data, **opt_kwargs)
    return plot_deviation_heatmap(ag_dist[0], ag_dist[1], HIgGs, data.Antigen.values)