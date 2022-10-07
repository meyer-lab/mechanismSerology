from ..preprocess import prepare_data, makeRcpAgLabels
from ..core import *
from .common import *
import numpy as np
import seaborn as sns
import xarray as xr 
import jax.numpy as jnp

def plotPrediction(data: xr.DataArray, lbound, ax=None):
    """
    Configures settings and creates scatterplot for before_after_optimize().
    """
    assert data.shape == lbound.shape
    cube_flat = data.values.flatten()
    valid = np.isfinite(cube_flat)
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
    final_f = plotPrediction(cube, new_lbound, axs[1])
    final_f.set_title("After Abundance Fit", fontsize=13)
    #add_r_text(cube, init_lbound, final_lbound, per_receptor, f)



def add_r_text(data, init_lbound, new_lbound, per_receptor, f):
    """
    Prints r value of each receptor and the average r value on the plot f.
    """



    ####################

    data_flat = (prepare_data(data)).values.flatten()
    nonzero = np.nonzero(data_flat)
    data_flat = data_flat[nonzero]
    lbound_flat_initial = init_lbound.flatten()[nonzero]
    lbound_flat_final = new_lbound.flatten() [nonzero]

    receptor_labels, ag_labels = makeRcpAgLabels(data)
    r_index_list = get_indices(data, per_receptor)
    labels = receptor_labels if per_receptor else ag_labels

    initial_r = calculate_r_list_from_index(data_flat, lbound_flat_initial, r_index_list)
    final_r = calculate_r_list_from_index(data_flat, lbound_flat_final, r_index_list)
    r_tot_initial = jnp.corrcoef(data_flat, lbound_flat_initial) [0,1]
    r_tot_final = jnp.corrcoef(data_flat, lbound_flat_final) [0,1]
    
    # initial
    start = 0.78
    for i in range(len(np.unique(labels))):
        f.text(0.05, start, '$r_{' + np.unique(labels)[i] + '}$' + r'= {:.2f}'.format(initial_r[i]), fontsize=12)
        start -=.03
    f.text(0.05, 0.86, '$r_{avg}$' + r'= {:.2f}'.format(sum(initial_r)/len(initial_r)), fontsize=12)
    f.text(0.05, 0.83, '$r_{total}$' + r'= {:.2f}'.format(r_tot_initial), fontsize=12)

    # final
    start = 0.78
    for i in range(len(np.unique(labels))):
        f.text(0.55, start, '$r_{' + np.unique(labels)[i] + '}$' + r'= {:.2f}'.format(final_r[i]), fontsize=12)
        start -=.03
    f.text(0.55, 0.86, '$r_{avg}$' + r'= {:.2f}'.format(sum(final_r)/len(final_r)), fontsize=12)
    f.text(0.55, 0.83, '$r_{total}$' + r'= {:.2f}'.format(r_tot_final), fontsize=12)