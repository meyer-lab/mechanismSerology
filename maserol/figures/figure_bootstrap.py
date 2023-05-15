"""
This is an internal figure which does not necessarily have a place in final
paper.
"""
import matplotlib.pyplot as plt

from tensordata.zohar import data as zohar

from maserol.heatmap import plot_deviation_heatmap
from maserol.preprocess import HIgGs, prepare_data
from maserol.resample import bootstrap

def makeFigure():
    return plt.figure()
    cube = prepare_data(zohar())
    ab_types = HIgGs
    post_opt_factor = True

    opt_kwargs = {
        "lrank": not post_opt_factor,
        "fitKa": False,
        "maxiter": 1000, # increase this to reduce inter-run variance
        "ab_types": ab_types,
        "post_opt_factor": post_opt_factor,
    }
    samp_dist, ag_dist = bootstrap(cube, numResample=3, norm="max", **opt_kwargs)
    f = plot_deviation_heatmap(ag_dist[0], ag_dist[1], ab_types, cube.Antigen.values)
    plt.title("Zohar Bootstrap With Post-opt Factorization")
    return f


