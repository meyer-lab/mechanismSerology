import jax.numpy as jnp
import matplotlib
import numpy as np
import random
import xarray as xr
from typing import Collection, List, Optional, Union

from maserol.core import DEFAULT_FIT_KA_VAL, DEFAULT_LRANK_VAL, inferLbound, optimizeLoss, reshapeParams
from .figure_utils import *
from .preprocess import assembleKav, makeRcpAgLabels

def configure_scatterplot(original_flat, pred_flat, receptors=False, antigens=False, loc=None, palette=None): 
    """
    Creates scatterplot and labels.
    """
    f = sns.scatterplot(x=original_flat, y=pred_flat, hue=receptors, style=antigens or None, ax=loc, s=70, alpha=0.5, palette=palette)
    f.set_xlabel("Actual", fontsize=12)
    f.set_ylabel("Predictions", fontsize=12)
    legend_label = "Receptor | Antigen" if antigens else "Receptor"
    f.legend(title=legend_label, bbox_to_anchor=(1, 1), borderaxespad=0)
    return f

def plot_lbound_correlation(data: xr.DataArray, lbound: Union[xr.DataArray, np.ndarray], rec: Union[Collection[str], str], ax=None, palette: Optional[List[str]] = None) -> matplotlib.axes.Axes:
    """
    Plots the lbound predictions vs their actual values on a scatter plot.

    Args:
        data: Prepared data as DataArray
        lbound: Predicted lbound as numpy array or DataArray
        rec: Receptor(s) to plot
        ax: Pre-existing axes for the plot
    
    Returns:
      matplotlib axes for the created plot
    """
    if isinstance(rec, str):
        rec = [rec]
    idx = np.where(np.isin(data.Receptor.values, rec))
    data_filtered = data.isel(Receptor=idx[0])
    data_filtered_flat = np.concatenate([np.ravel(t) for t in np.split(data_filtered.values, data_filtered.values.shape[1], axis=1)])
    lbound_filtered = lbound[:, idx[0], :]
    lbound_filtered_flat = np.concatenate([np.ravel(t) for t in np.split(lbound_filtered, lbound_filtered.shape[1], axis=1)])
    assert lbound_filtered_flat.shape == data_filtered_flat.shape
    n_points_per_rec = lbound_filtered_flat.shape[0] / len(rec)
    assert n_points_per_rec.is_integer()
    n_points_per_rec = int(n_points_per_rec)
    labels = np.concatenate([np.full(n_points_per_rec, r) for r in data.Receptor.values[idx[0]]])
    f = sns.scatterplot(x=np.log(data_filtered_flat), y=np.log(lbound_filtered_flat), hue=labels, alpha=0.8, ax=ax, palette=palette)
    f.set_xlabel("Actual", fontsize=12)
    f.set_ylabel("Predictions", fontsize=12)
    f.legend(title="Receptor", bbox_to_anchor=(1, 1), borderaxespad=0)
    return f

def leave_out_rec(data: xr.DataArray, rec: Union[Collection[str], str], **opt_kwargs) -> np.ndarray:
    """
    Trains the model, leaving out the receptors specified by rec.

    Args:
        data: prepared data in DataArray form
        rec: collection or individual receptor as a string

    Returns:
        Inferred lbound for all receptors specified in data.
    """
    if isinstance(rec, str):
        rec = [rec]
    idx = np.where(np.logical_not(np.isin(data.Receptor.values, rec)))
    data_no_rec = data.isel(Receptor=idx[0])
    opt_x, _ = optimizeLoss(data_no_rec, **opt_kwargs)
    infer_lbound_kwargs = {k: v for k, v in opt_kwargs.items() if k in ("lrank", "fitKa")}
    params = reshapeParams(opt_x, data, **infer_lbound_kwargs)
    if not opt_kwargs.get("fitKa", DEFAULT_FIT_KA_VAL):
        params.append(assembleKav(data).values)
    lbound = inferLbound(data, *params, lrank=opt_kwargs.get("lrank", DEFAULT_LRANK_VAL))
    return lbound

def plot_leave_out_rec_lbound_correlation(data: Union[xr.DataArray, np.ndarray], rec: Union[Collection[str], str], **opt_kwargs) -> matplotlib.axes.Axes:
    """
    Trains the model on data that excludes receptor(s) specified by rec. Plots
    the correlation between predicted and actual data as a scatter plot as 3 plots:
        1. All minus excluded receptors
        2. Excluded receptors
        3. All
    
    Args:
        data: Prepared data as DataArray or np array
        rec: receptor(s) to leave out

    Returns:
        axes for plot
    """
    if isinstance(rec, str):
        rec = [rec]
    lbound = leave_out_rec(data, rec, **opt_kwargs) 

    palette = {r: (random.random(), random.random(), random.random()) for r in data.Receptor.values}

    axes, plot = getSetup((20, 6), (1, 3))
    all_minus_rec = [r for r in data.Receptor.values if r not in rec]

    f = plot_lbound_correlation(data, lbound, all_minus_rec, ax=axes[0], palette=palette)
    f.set_title(f"All - {', '.join(rec)}", fontsize=15)

    f = plot_lbound_correlation(data, lbound, rec, ax=axes[1], palette=palette)
    f.set_title(f"{', '.join(rec)}", fontsize=15)

    f = plot_lbound_correlation(data, lbound, data.Receptor.values, ax=axes[2], palette=palette)
    f.set_title(f"All", fontsize=15)

    return plot

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


 