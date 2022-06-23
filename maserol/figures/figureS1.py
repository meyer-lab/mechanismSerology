from tensorpack.cmtf import perform_CP
from tensorpack.plot import plot_weight_mode
from .common import getSetup
from ..linear import importConcat

def makeFigure():
    axs, f = getSetup((8, 3), (1, 3))
    combined, M, S, _ = importConcat()

    tensor = combined.values
    tfac = perform_CP(tOrig=tensor, r=6)
    for ii, ax in enumerate(axs):
        if (ii != 0):
            label = combined.coords[combined.dims[ii]].values
        else:
            label = False
        plot_weight_mode(ax, tfac.factors[ii], labels=label, title=combined.dims[ii])

    axs[0].set_yticks([0, S['Sample'].size, S['Sample'].size + M['Sample'].size])
    axs[0].set_yticklabels(["SpaceX", "MGH", "Zohar"])

    return f