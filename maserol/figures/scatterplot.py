from ..preprocess import prepare_data, makeRcpAgLabels
from ..core import *
from .common import *
import numpy as np
import seaborn as sns
import xarray as xr 
import jax.numpy as jnp

def plotPrediction(data: xr.DataArray, lbound, ax=None):
    """ Create a basic figure of actual vs predict scatterplot. """
    assert data.shape == lbound.shape
    cube_flat = data.values.flatten()
    valid = cube_flat > 0
    receptor_labels, antigen_labels = makeRcpAgLabels(data)

    # plot
    f = sns.scatterplot(x=np.log(cube_flat[valid]),
                        y=np.log(lbound.flatten()[valid]),
                        hue=receptor_labels[valid],
                        style=antigen_labels[valid],
                        ax=ax, s=70, alpha=0.5)
    f.legend(title="Receptor | Antigen", bbox_to_anchor=(1, 1), borderaxespad=0)
    f.set_xlabel("Actual", fontsize=12)
    f.set_ylabel("Predictions", fontsize=12)
    return f

def plotOptimize(data: xr.DataArray, metric="mean", lrank=True, fitKa=False,
                 n_ab=1, maxiter=500, fucose=False):
    """ Run optimizeLoss(), and compare scatterplot before and after """
    cube = prepare_data(data)
    x_opt, opt_f, init_p = optimizeLoss(cube, metric=metric, lrank=lrank, fitKa=fitKa,
                                        n_ab=n_ab, maxiter=maxiter, fucose=fucose, retInit=True)

    init_lbound = inferLbound(cube, *init_p, lrank=lrank, L0=1e-9, KxStar=1e-12)

    new_p = reshapeParams(x_opt, cube, lrank=lrank, fitKa=fitKa)
    if not fitKa:
        new_p.append(init_p[-1])
    new_lbound = inferLbound(cube, *new_p, lrank=lrank, L0=1e-9, KxStar=1e-12)

    if metric == "mean":
        init_lbound *= INIT_SCALER
        new_lbound *= x_opt[-1]
        print("Scalar", x_opt[-1])

    axs, f = getSetup((13, 5), (1, 2))
    sns.set(style="darkgrid", font_scale=1)
    initial_f = plotPrediction(cube, init_lbound, axs[0])
    initial_f.set_title("Initial", fontsize=13)
    new_f = plotPrediction(cube, new_lbound, axs[1])
    new_f.set_title("After Abundance Fit", fontsize=13)

    # Add R numbers onto plot
    Raxis = -1
    if metric == "rrcp":
        Raxis = 1
    if metric == "rag":
        Raxis = 2
    f.text(0.05, 0.1, gen_R_labels(cube, init_lbound, Raxis), fontsize=12)
    f.text(0.55, 0.1, gen_R_labels(cube, new_lbound, Raxis), fontsize=12)
    return f


def gen_R_labels(cube, lbound, axis=-1):
    """ Make a long string on the R breakdowns for plotting purpose """
    retstr = ""
    r_tot = calcModalR(cube, lbound, axis=-1, valid_idx=getNonnegIdx(cube, metric="rtot"))
    retstr += '$r_{total}$' + r'= {:.2f}'.format(r_tot) + '\n'
    if axis > 0:
        r_labels = cube.Receptor.values if axis == 1 else cube.Antigen.values
        r_s = calcModalR(cube, lbound, axis=axis,
                         valid_idx=getNonnegIdx(cube, metric=("rrcp" if axis==1 else "rag")))
        retstr += '$r_{avg}$' + r'= {:.2f}'.format(sum(r_s)/len(r_s)) + '\n'
        for ii in range(len(r_labels)):
            retstr += '$r_{' + r_labels[ii] + '}$' + r'= {:.2f}'.format(r_s[ii]) + '\n'
    return retstr
