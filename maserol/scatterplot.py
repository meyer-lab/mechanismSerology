import copy
from typing import Collection, Mapping, Optional, Union, List

import xarray
import numpy as np
import matplotlib
import seaborn as sns
from .preprocess import make_rcp_ag_labels, HIgGs, assemble_Ka, prepare_data
from .core import (
    infer_Lbound,
    reshape_params,
    optimize_loss,
    DEFAULT_AB_TYPES,
    DEFAULT_FIT_KA_VAL,
)
from .figures.common import getSetup


def plotPrediction(data: xarray.DataArray, lbound, ax=None, logscale=True):
    """Create a basic figure of actual vs predict scatterplot."""
    assert data.shape == lbound.shape
    cube_flat = data.values.flatten()

    valid_idx = np.where(cube_flat > 0)
    receptor_labels, antigen_labels = make_rcp_ag_labels(data)

    x = cube_flat[valid_idx]
    y = lbound.flatten()[valid_idx]
    if logscale:
        x = np.log(x)
        y = np.log(y)

    # plot
    f = sns.scatterplot(
        x=x,
        y=y,
        hue=receptor_labels[valid_idx],
        # style=antigen_labels[valid_idx],
        ax=ax,
        s=70,
        alpha=0.5,
    )
    f.legend(title="Receptor | Antigen", bbox_to_anchor=(1, 1), borderaxespad=0)
    f.set_xlabel("Actual", fontsize=12)
    f.set_ylabel("Predictions", fontsize=12)
    return f


def plotOptimize(
    data: xarray.DataArray,
    L0: np.ndarray,
    KxStar: np.ndarray,
    f: np.ndarray,
    ab_types: tuple[str] = HIgGs,
):
    """Run optimizeLoss(), and compare scatterplot before and after"""
    cube = prepare_data(data)
    Ka = assemble_Ka(cube, ab_types).values
    x_opt, ctx = optimize_loss(cube, L0, KxStar, f, ab_types=ab_types)

    init_lbound = infer_Lbound(cube, ctx["init_params"][0], Ka, L0, KxStar, f)

    new_p = reshape_params(x_opt, cube, ab_types=ab_types)
    final_lbound = infer_Lbound(cube, new_p[0], Ka, L0, KxStar, f)

    init_lbound = np.log10(init_lbound)
    final_lbound = np.log10(final_lbound)

    axs, f = getSetup((13, 5), (1, 2))
    cube_log = np.log10(cube)
    sns.set(style="darkgrid", font_scale=1)
    initial_f = plotPrediction(cube_log, init_lbound, axs[0], logscale=False)
    initial_f.set_title("Initial", fontsize=13)
    new_f = plotPrediction(cube_log, final_lbound, axs[1], logscale=False)
    new_f.set_title("After Abundance Fit", fontsize=13)

    # Add R numbers onto plot
    cube_flat = cube_log.values.flatten()
    valid_idx = np.where(cube_flat > 0)
    f.text(
        0.05,
        0.1,
        gen_R_labels(cube_flat[valid_idx], init_lbound.flatten()[valid_idx]),
        fontsize=12,
    )
    f.text(
        0.55,
        0.1,
        gen_R_labels(cube_flat[valid_idx], final_lbound.flatten()[valid_idx]),
        fontsize=12,
    )
    return f


def gen_R_labels(cube_flat, lbound_flat):
    """Make a long string on the R breakdowns for plotting purpose"""
    retstr = ""
    rtot = np.corrcoef(cube_flat, lbound_flat)[0, 1]
    retstr += "$r_{total}$" + r"= {:.2f}".format(rtot) + "\n"
    return retstr


def plotLbound(
    data: xarray.DataArray,
    lbound: Union[xarray.DataArray, np.ndarray],
    rec: Union[Collection[str], str],
    ax=None,
    palette: Optional[Union[List, Mapping]] = None,
) -> matplotlib.axes.Axes:
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
    data_filtered_flat = np.concatenate(
        [
            np.ravel(t)
            for t in np.split(
                data_filtered.values, data_filtered.values.shape[1], axis=1
            )
        ]
    )
    lbound_filtered = lbound[:, idx[0], :]
    lbound_filtered_flat = np.concatenate(
        [
            np.ravel(t)
            for t in np.split(lbound_filtered, lbound_filtered.shape[1], axis=1)
        ]
    )
    assert lbound_filtered_flat.shape == data_filtered_flat.shape
    n_points_per_rec = lbound_filtered_flat.shape[0] / len(rec)
    assert n_points_per_rec.is_integer()
    n_points_per_rec = int(n_points_per_rec)
    labels = np.concatenate(
        [np.full(n_points_per_rec, r) for r in data.Receptor.values[idx[0]]]
    )
    f = sns.scatterplot(
        x=np.log(data_filtered_flat),
        y=np.log(lbound_filtered_flat),
        hue=labels,
        alpha=0.8,
        ax=ax,
        palette=palette,
    )
    f.set_xlabel("Actual", fontsize=12)
    f.set_ylabel("Predictions", fontsize=12)
    f.legend(title="Receptor", bbox_to_anchor=(1, 1), borderaxespad=0)
    return f


def LRcpO(
    data: xarray.DataArray, rec: Union[Collection[str], str], **opt_kwargs
) -> np.ndarray:
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
    # recompute FcIdx with the remaining receptors
    data_no_rec = data.isel(Receptor=idx[0])
    opt_kwargs_sub = copy.deepcopy(opt_kwargs)
    opt_kwargs_sub["L0"] = opt_kwargs_sub["L0"][idx]
    opt_kwargs_sub["KxStar"] = opt_kwargs_sub["KxStar"][idx]
    opt_kwargs_sub["f"] = opt_kwargs_sub["f"][idx]
    opt_x, _ = optimize_loss(
        data_no_rec, **opt_kwargs_sub, ftol=1e-7, xtol=1e-7, gtol=1e-7
    )
    ab_types = opt_kwargs_sub.get("ab_types", DEFAULT_AB_TYPES)
    Rtot = reshape_params(opt_x, data, ab_types=ab_types)[0]
    Ka = assemble_Ka(data, ab_types).values
    lbound = infer_Lbound(
        data, Rtot, Ka, opt_kwargs["L0"], opt_kwargs["KxStar"], opt_kwargs["f"]
    )
    return lbound


def plotLRcpO(
    data: xarray.DataArray, rec: Union[Collection[str], str], **opt_kwargs
) -> matplotlib.axes.Axes:
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
    lbound = LRcpO(data, rec, **opt_kwargs)

    palette_list = sns.color_palette("bright", data.Receptor.values.shape[0])
    palette = {r: color for r, color in zip(data.Receptor.values, palette_list)}

    axes, plot = getSetup((20, 6), (1, 3))
    all_minus_rec = [r for r in data.Receptor.values if r not in rec]

    f = plotLbound(data, lbound, all_minus_rec, ax=axes[0], palette=palette)
    f.set_title(f"All - {', '.join(rec)}", fontsize=15)

    f = plotLbound(data, lbound, rec, ax=axes[1], palette=palette)
    f.set_title(f"{', '.join(rec)}", fontsize=15)

    f = plotLbound(data, lbound, data.Receptor.values, ax=axes[2], palette=palette)
    f.set_title(f"All", fontsize=15)

    return plot
