"""
This is an internal figure which does not necessarily have a place in final
paper.
"""
from tensordata.zohar import data as zohar

from maserol.preprocess import assemble_options, HIgGs, prepare_data
from maserol.scatterplot import plotOptimize


def makeFigure():
    cube_zohar = prepare_data(zohar())
    cube_zohar = cube_zohar[: cube_zohar.shape[0] // 10]
    opts = assemble_options(cube_zohar, ab_types=("IgG1", "IgG2", "IgG3"))
    f = plotOptimize(cube_zohar, opts)
    f.text(0.35, 1, "Zohar Optimization")
    return f
