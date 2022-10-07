from ..preprocess import prepare_data
from .common import *
import numpy as np
import seaborn as sns
import xarray as xr 
import jax.numpy as jnp

def configure_scatterplot(data: xr.DataArray, lbound, loc=None):
    """
    Configures settings and creates scatterplot for make_initial_final_lbound_correlation_plot.
    """
    # prepare data
    cube_flat = data.values.flatten()
    print(cube_flat.min())
    nonzero = np.nonzero(cube_flat)
    print(nonzero)
    receptor_labels, antigen_labels = make_rec_subj_labels(data)

    lbound_flat = lbound.flatten()[nonzero]
    cube_flat = cube_flat[nonzero]
    receptor_labels = receptor_labels[nonzero]
    antigen_labels = antigen_labels[nonzero]

    # plot
    f = sns.scatterplot(x=np.log(lbound_flat), y=np.log(cube_flat), hue=receptor_labels, style=antigen_labels, ax=loc, s=70, alpha=0.5)
    f.legend(title="Receptor | Antigen", bbox_to_anchor=(1, 1), borderaxespad=0)
    f.set_xlabel("Predictions", fontsize=12)
    f.set_ylabel("Actual", fontsize=12)
    return f

def make_initial_final_lbound_correlation_plot(cube, initial_lbound, final_lbound, per_receptor):
    """
    Creates two scatterplots comparing correlations between the cube and the initial and final lbound predictions.
    """
    axs, f = getSetup((13, 5), (1, 2))
    sns.set(style="darkgrid", font_scale=1)
    initial_f = configure_scatterplot(cube, initial_lbound, axs[0])
    initial_f.set_title("Initial", fontsize=13)
    final_f = configure_scatterplot(cube, final_lbound, axs[1])
    final_f.set_title("After Abundance Fit", fontsize=13)
    add_r_text(cube, initial_lbound, final_lbound, per_receptor, f)
    return f

def add_r_text(cube, initial_lbound, final_lbound, per_receptor, f):
    """
    Prints r value of each receptor and the average r value on the plot f.
    """
    cube_flat = (prepare_data(cube)).values.flatten()
    nonzero = np.nonzero(cube_flat)
    cube_flat = cube_flat[nonzero]
    lbound_flat_initial = initial_lbound.flatten()[nonzero]
    lbound_flat_final = final_lbound.flatten() [nonzero]

    receptor_labels, ag_labels = make_rec_subj_labels(cube)
    r_index_list = get_indices(cube, per_receptor)
    labels = receptor_labels if per_receptor else ag_labels

    initial_r = calculate_r_list_from_index(cube_flat, lbound_flat_initial, r_index_list)
    final_r = calculate_r_list_from_index(cube_flat, lbound_flat_final, r_index_list)
    r_tot_initial = jnp.corrcoef(cube_flat, lbound_flat_initial) [0,1]
    r_tot_final = jnp.corrcoef(cube_flat, lbound_flat_final) [0,1]
    
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