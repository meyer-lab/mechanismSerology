from .common import getSetup, subplotLabel
from ..predictAbundKa import optimize_lossfunc, plot_correlation_heatmap
from tensordata.atyeo import data as atyeo

def makeFigure():
    axs, f = getSetup((8, 4), (1, 2))

    d = atyeo()
    cube = d.tensor
    _, rec_names, ant_names = d.axes
    RKa_opt = optimize_lossfunc(cube, n_ab=1, maxiter=1000)

    # heatmap for correlation
    plot_correlation_heatmap(axs[0], RKa_opt, cube, rec_names, ant_names)

    subplotLabel(axs)
    return f


