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
    cube = prepare_data(zohar())
    ab_types = HIgGs
    post_opt_factor = True

    opt_kwargs = {
        "metric": "mean_rcp",
        "lrank": not post_opt_factor,
        "fitKa": False,
        "ab_types": ab_types,
        "maxiter": 1000,
        "post_opt_factor": post_opt_factor,
    }
    samp_dist, ag_dist = bootstrap(cube, numResample=3, norm="max", **opt_kwargs)
    return plot_deviation_heatmap(ag_dist[0], ag_dist[1], ab_types, cube.Antigen.values)
