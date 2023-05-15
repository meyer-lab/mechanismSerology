"""
This is an internal figure which does not necessarily have a place in final
paper.
"""
from tensordata.zohar import data as zohar

from maserol.preprocess import HIgGs, prepare_data
from maserol.scatterplot import plotOptimize

def makeFigure():
    cube_zohar = prepare_data(zohar())
    f = plotOptimize(cube_zohar, fitKa=False, ab_types=HIgGs)
    f.text(0.35, 1, "RMSE Optimization with Single Scaling Factor (Zohar)")
    return f
