"""
This creates Figure 0.
"""
import numpy as np
from scipy.optimize import least_squares
from tensordata.alter import data as alter

from .common import subplotLabel, getSetup


def pfunc(x, p):
    return np.power(x, p[0]) * p[1]


def makeFigure():
    """ Compare genotype vs non-genotype specific readings. """
    data = alter().Fc
    data = data.stack(SAg=("Sample", "Antigen"))
    receptors = data.coords["Receptor"].values

    axs, fig = getSetup((8, 8), (3, 3))

    idxsa = np.array([1, 1, 2, 4, 4, 5, 7, 8, 9])
    idxsb = np.array([2, 3, 3, 5, 6, 6, 6, 7, 8])

    for ii, ax in enumerate(axs):
        xx = data[idxsa[ii], :]
        yy = data[idxsb[ii], :]

        ax.scatter(xx, yy, s=0.3)

        popt = least_squares(lambda x: np.nan_to_num(pfunc(xx, x) - yy), x0=[1.0, 1.0], jac="3-point")
        linx = np.linspace(0.0, np.amax(xx), num=100)
        liny = pfunc(linx, popt.x)
        ax.plot(linx, liny, "r-")

        ax.set_xlabel(receptors[idxsa[ii]])
        ax.set_ylabel(receptors[idxsb[ii]])
        ax.set_xticks(ax.get_xticks().tolist())
        ax.set_xticklabels(ax.get_xticks().tolist(), rotation=20, ha="right")
        ax.set_ylim(bottom=-2000, top=180000)
        ax.set_xlim(left=-2000, right=180000)

    subplotLabel(axs)

    return fig
