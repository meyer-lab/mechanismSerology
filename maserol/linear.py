from tensordata.kaplonek import MGH, MGH4D, SpaceX
from tensordata.zohar import data3D
from tensorpack.decomposition import Decomposition
from matplotlib.ticker import ScalarFormatter
import numpy as np

def checkMissingess(cube):
    return 1 - np.sum(np.isfinite(cube))/np.product(cube.shape)


def normalizeSubj(cube):
    cube -= np.nanmean(cube, axis=0)
    cube = cube / np.nanstd(cube, axis=0)
    return cube


""" Assemble the concatenated COVID tensor in 3D """
def importConcat():
    M, S, Z = MGH(xarray=True), SpaceX(xarray=True), data3D(xarray=True)

    M = normalizeSubj(M)
    S = normalizeSubj(S)
    Z = normalizeSubj(Z)

    # Set consistent antigen naming
    M_dict = {'Antigen': ['SARS.CoV2_N', 'CoV.OC43', 'Flu_HA', 'SARS.CoV2_S1', 'Ebola_gp', 'CMV',
                                          'SARS.CoV2_S', 'SARS.CoV2_S2', 'SARS.CoV2_RBD']}

    S_dict = {'Antigen': ['CoV.HKU1_S', 'CoV.OC43_RBD', 'CoV.HKU1_RBD', 'CoV.OC43_S', 'SARS.CoV2_S',
                            'SARS.CoV2_S1', 'SARS.CoV2_RBD', 'SARS_RBD', 'SARS.CoV2_S2', 'Flu_HA',
                            'Ebola_gp', 'MERS_RBD', 'SARS_S', 'MERS_S'],
                'Sample': S['Sample'].values.astype(str)}

    Z_dict = {'Antigen': ['SARS.CoV2_S', 'SARS.CoV2_RBD', 'SARS.CoV2_N', 'SARS.CoV2_S1','SARS.CoV2_S2', 
                            'SARS.CoV2_S1trimer']}

    M, S, Z = M.assign_coords(M_dict), S.assign_coords(S_dict), Z.assign_coords(Z_dict)
    cube = S.combine_first(M).combine_first(Z)
    cube = normalizeSubj(cube)

    return cube, M, S, Z

def importConcat4D():
    pass

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