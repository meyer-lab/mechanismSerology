from tensordata.zohar import data as zohar

from maserol.preprocess import assemble_options, prepare_data
from maserol.scatterplot import plot_optimize


def makeFigure():
    cube_zohar = prepare_data(zohar())
    opts = assemble_options(cube_zohar)
    opts["tol"] = 1e-5
    return plot_optimize(cube_zohar, opts)[0]
