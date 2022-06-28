from matplotlib import gridspec, pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorpack.decomposition import Decomposition
from matplotlib.ticker import ScalarFormatter


def makeComponentPlot(comps, axes):
    rank = comps[0].shape[1]
    components = [str(ii + 1) for ii in range(rank)]

    subs = pd.DataFrame(comps[0], columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)], index=axes[0])
    rec = pd.DataFrame(comps[1], columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)], index=axes[1])
    ant = pd.DataFrame(comps[2], columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)], index=axes[2])

    f = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.5)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    sns.heatmap(subs, cmap="PiYG", center=0, xticklabels=components, yticklabels=axes[0], cbar=True, vmin=-1.0, vmax=1.0, ax=ax1)
    sns.heatmap(rec, cmap="PiYG", center=0, xticklabels=components, yticklabels=axes[1], cbar=False, vmin=-1.0, vmax=1.0, ax=ax2)
    sns.heatmap(ant, cmap="PiYG", center=0, xticklabels=components, yticklabels=axes[2], cbar=False, vmin=-1.0, vmax=1.0, ax=ax3)

    ax1.set_xlabel("Components")
    ax1.set_title("Subjects")
    ax2.set_xlabel("Components")
    ax2.set_title("Receptors")
    ax3.set_xlabel("Components")
    ax3.set_title("Antigens")
    return f

def compareR2X(ax, decomp1:Decomposition, decomp2:Decomposition, label1="", label2=""):
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
    
    assert(decomp1.rrs.size == decomp2.rrs.size)
    
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
    CPR2X1, CPR2X2, sizeTfac1, sizeTfac2 = np.asarray(decomp1.TR2X), np.asarray(decomp2.TR2X), decomp1.sizeT, decomp2.sizeT
    ax.set_xscale("log", base=2)
    ax.plot(sizeTfac1, 1.0 - CPR2X1, ".", label=label1)
    ax.plot(sizeTfac2, 1.0 - CPR2X2, ".", label=label2)
    ax.set_ylabel("Normalized Unexplained Variance")
    ax.set_xlabel("Size of Reduced Data")
    ax.set_title("Tfac Data Reduction")
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()
