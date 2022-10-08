from .common import *
from ..preprocess import makeRcpAgLabels
import numpy as np
import xarray as xr
import jax.numpy as jnp

def configure_scatterplot(original_flat, pred_flat, receptors=False, antigens=False, loc=None, palette=None): 
    """
    Creates scatterplot and labels.
    """
    f = sns.scatterplot(x=original_flat, y=pred_flat, hue=receptors, style=antigens, ax=loc, s=70, alpha=0.5, palette=palette)
    f.set_xlabel("Actual", fontsize=12)
    f.set_ylabel("Predictions", fontsize=12)
    legend_label = "Receptor | Antigen" if len(antigens) != 0 else "Receptor"
    f.legend(title=legend_label, bbox_to_anchor=(1, 1), borderaxespad=0)
    return f

def plot_receptor_validation_plot(receptor, data : xr.DataArray, cube_flat, lbound_flat, colors, antigens=False):
    '''
    Plots three scatterplots sumarizing results of receptor crossvalidation: (1) the receptor correlation, (2) total correlation,
    (3) correlation without the receptor.
    Inputs:
        receptor (str): the receptor to compare
        data (DataArray):
        cube_flat: values in data as a vector
        lbound_flat: predicted binding without receptor as a vector
        colors: list of colors with the same number of colors as original number of receptors in 'data'
        antigens (bool): Set scatterplot hue to different shapes representing antigens when True, disreagrd when False.
    '''
    r_list = []
    non_nan = ~np.isnan(cube_flat)
    cube_flat = cube_flat[non_nan]

    receptor_labels, antigen_labels = makeRcpAgLabels(data)
    receptor_labels = receptor_labels[non_nan]
    antigen_labels = antigen_labels[non_nan] if antigens else None
    lbound_flat = lbound_flat[non_nan]
    axes, plt = getSetup((20,6), (1,3))
    
    # coordinate color scheme
    c_index = list(data.Receptor.values).index(receptor)
    r_color = [str(colors[c_index]),]
    r_color_sub = colors[:c_index] + colors[c_index+1:]
    indices = np.where(receptor_labels == receptor)
    
    # RECEPTOR
    g = configure_scatterplot(cube_flat[indices], lbound_flat[indices], receptors=receptor_labels[indices], antigens=antigen_labels[indices], loc=axes[0], palette=r_color)
    r_list.append(jnp.corrcoef(cube_flat[indices], lbound_flat[indices]) [0,1])
    g.set_title(receptor, fontsize=15)

    # All 
    f = configure_scatterplot(cube_flat, lbound_flat, receptors=receptor_labels, antigens=antigen_labels, loc=axes[1], palette=colors)
    r_list.append(jnp.corrcoef(cube_flat, lbound_flat) [0,1])
    f.set_title("All", fontsize=15)
    
    # ALL - RECEPTOR
    non_indices = np.where(receptor_labels != receptor)
    h = configure_scatterplot(cube_flat[non_indices], lbound_flat[non_indices], receptors=receptor_labels[non_indices], antigens=antigen_labels[non_indices], loc=axes[2], palette=r_color_sub)
    r_list.append(jnp.corrcoef(cube_flat[non_indices], lbound_flat[non_indices]) [0,1])
    h.set_title(f"All - {receptor}", fontsize=15) 

    return plt, r_list

def make_initial_final_lbound_correlation_plot(data : xr.DataArray, cube_flat, ilbound, flbound, per_receptor=True, antigens=True):
    """
    Creates two scatterplots comparing correlations between the cube and the initial and final lbound predictions.
    Inputs:
        data (DataArray):
        cube_flat: values in data as a vector.
        i/flbound: initial/final predicted binding as a vector.
        per_receptors (bool): calcualtes r per-receptor when True, calculated per-antigen when False
        antigens (bool): Set scatterplot hue to different shapes representing antigens when True, disreagrd when False.
    """
    # prepare data
    non_nan = ~np.isnan(cube_flat)
    cube_flat = cube_flat[non_nan]

    receptor_labels, antigen_labels = makeRcpAgLabels(data)
    antigen_labels = antigen_labels[non_nan] if antigens else None
    receptor_labels = receptor_labels[non_nan]
    ilbound = ilbound[non_nan]
    flbound = flbound[non_nan]
    
    axs, f = getSetup((13, 5), (1, 2))
    initial_f = configure_scatterplot(cube_flat, ilbound, receptors=receptor_labels, antigens=antigen_labels, loc=axs[0], palette=None)
    initial_f.set_title("Initial", fontsize=13)
    final_f = configure_scatterplot(cube_flat, flbound, receptors=receptor_labels, antigens=antigen_labels, loc=axs[1], palette=None)
    final_f.set_title("After Abundance Fit", fontsize=13)
    add_r_text(data, ilbound, flbound, per_receptor, f, antigens)
    return f

def add_r_text(cube, ilbound, flbound, per_receptor, f, antigens=False):
    """
    Prints r value of each receptor and the average r value on the plot f.
    """
    cube_flat = cube.values.flatten()
    non_nan = ~np.isnan(cube_flat)
    cube_flat = cube_flat[non_nan]
    receptor_labels, ag_labels = makeRcpAgLabels(cube)
    r_index_list = get_indices(cube, per_receptor)
    labels = receptor_labels if per_receptor else ag_labels

    r_tot_initial = jnp.corrcoef(cube_flat, ilbound) [0,1]
    r_tot_final = jnp.corrcoef(cube_flat, flbound) [0,1]
    f.text(0.06, 0.83, '$r_{total}$' + r'= {:.2f}'.format(r_tot_initial), fontsize=12)
    f.text(0.55, 0.83, '$r_{total}$' + r'= {:.2f}'.format(r_tot_final), fontsize=12)
    if antigens:
        initial_r = calculate_r_list_from_index(cube_flat, ilbound, r_index_list)
        final_r = calculate_r_list_from_index(cube_flat, flbound, r_index_list)
        
        # initial
        start = 0.78
        for i in range(len(np.unique(labels))):
            f.text(0.05, start, '$r_{' + np.unique(labels)[i] + '}$' + r'= {:.2f}'.format(initial_r[i]), fontsize=12)
            start -=.03
        f.text(0.05, 0.86, '$r_{avg}$' + r'= {:.2f}'.format(sum(initial_r)/len(initial_r)), fontsize=12)

        # final
        start = 0.78
        for i in range(len(np.unique(labels))):
            f.text(0.55, start, '$r_{' + np.unique(labels)[i] + '}$' + r'= {:.2f}'.format(final_r[i]), fontsize=12)
            start -=.03
        f.text(0.55, 0.86, '$r_{avg}$' + r'= {:.2f}'.format(sum(final_r)/len(final_r)), fontsize=12)


 