"""
This is an internal figure which does not necessarily have a place in final
paper.
"""
from tensordata.zohar import data as zohar

from maserol.core import prepare_data
from maserol.preprocess import HIgGs
from maserol.scatterplot import plotOptimize

def makeFigure():
    cube_zohar = prepare_data(zohar())
    f = plotOptimize(cube_zohar, "mean_autoscale", lrank=False, fitKa=False, maxiter=500, ab_types=HIgGs)
    f.text(0.35, 1, "RMSE Optimization with Single Autoscale Term (Zohar)")
    return f