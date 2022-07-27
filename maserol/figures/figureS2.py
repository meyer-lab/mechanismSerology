from tensordata.kaplonek import MGH, MGH4D
from .common import getSetup
import numpy as np
from tensorpack.decomposition import Decomposition
from matplotlib.ticker import ScalarFormatter


def compareR2X(ax, decomp1: Decomposition, decomp2: Decomposition, label1="", label2=""):
    """
    Plots R2X for tensor factorizations for all components up to decomp.max_rr. Based off tfacr2x in tensorpack.plot.
    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f. See getSetup() in tensorpack.test.common.py for more detail.
    decomp1, decomp2 : Decomposition
        Takes two Decomposition objects that have successfully run decomp.perform_tfac() and have the same number of components.
    label1, label2 : string
        Takes names of plotted elements for the legend.
    """

    assert (decomp1.rrs.size == decomp2.rrs.size)

    comps = decomp1.rrs
    ax.scatter(comps, decomp1.TR2X, s=15, label=label1)
    ax.scatter(comps, decomp2.TR2X, s=15, label=label2)
    ax.set_ylabel("Tensor Fac R2X")
    ax.set_xlabel("Number of Components")
    ax.set_title("Variance explained by tensor decomposition")
    ax.set_xticks([x for x in comps])
    ax.set_xticklabels([x for x in comps])
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, np.amax(comps) + 0.5)
    ax.legend()


def compareReduction(ax, decomp1, decomp2, label1="Tfac 1", label2="Tfac 2"):
    """
    Plots size reduction for tensor factorizations for all components up to decomp.max_rr. Based off reduction in tensorpack.plot.
    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f.
    decomp1, decomp2 : Decomposition
        Takes two Decomposition objects that have successfully run decomp.perform_tfac().
    label1, label2 : string
        Takes names of plotted elements for the legend.
    """
    CPR2X1, CPR2X2, sizeTfac1, sizeTfac2 = np.asarray(decomp1.TR2X), np.asarray(
        decomp2.TR2X), decomp1.sizeT, decomp2.sizeT
    ax.set_xscale("log", base=2)
    ax.plot(sizeTfac1, 1.0 - CPR2X1, ".", label=label1)
    ax.plot(sizeTfac2, 1.0 - CPR2X2, ".", label=label2)
    ax.set_ylabel("Normalized Unexplained Variance")
    ax.set_xlabel("Size of Reduced Data")
    ax.set_title("Tfac Data Reduction")
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()

def makeFigure():

    ax, f = getSetup((8, 4), (1, 2))

    decomp3D = Decomposition(data=MGH().tensor, max_rr=10)
    decomp3D.perform_tfac()

    tensor4D = np.asarray(MGH4D().values).astype('float64')
    decomp4D = Decomposition(data=tensor4D, max_rr=10)
    decomp4D.perform_tfac()

    compareR2X(ax[0], decomp3D, decomp4D, "Kaplonek 3D", "Kaplonek 4D")
    compareReduction(ax[1], decomp3D, decomp4D, "Kaplonek 3D", "Kaplonek 4D")

    return f

