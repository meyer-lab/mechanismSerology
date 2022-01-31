from .common import getSetup, subplotLabel
from ..predictAbundKa import optimize_lossfunc, plot_correlation_heatmap
import pickle
from ..data.atyeo import createCube, getAxes

def makeFigure():
    axs, f = getSetup((8, 4), (1, 2))

    cube = createCube()
    _, rec_names, ant_names = getAxes()
    RKa_opt = optimize_lossfunc(cube, n_ab=1, maxiter=100)

    # heatmap for correlation
    plot_correlation_heatmap(axs[0], RKa_opt, cube, rec_names, ant_names)

    subplotLabel(axs)
    return f


