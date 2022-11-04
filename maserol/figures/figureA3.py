"""
This is an internal figure which does not necessarily have a place in final
paper.
"""
from tensordata.zohar import data as zohar

from maserol.core import prepare_data
from maserol.heatmap import plot_deviation_heatmap
from maserol.preprocess import HIgGs
from maserol.resample import bootstrap

def makeFigure():
    data = prepare_data(zohar())
    opt_kwargs = {
        "metric": "rtot",
        "lrank": True,
        "fitKa": False,
        "ab_types": HIgGs,
    }
    sample_dist, ag_dist = bootstrap(data, numResample=10, norm="max", **opt_kwargs)
    return plot_deviation_heatmap(ag_dist[0], ag_dist[1], HIgGs, data.Antigen.values)