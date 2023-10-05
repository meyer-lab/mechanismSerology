import copy
from typing import Collection, Mapping, Optional, Union, List, Dict

import xarray
import numpy as np
import matplotlib
import seaborn as sns
from .preprocess import assemble_Ka
from .core import (
    infer_Lbound,
    infer_Lbound_mv,
    reshape_params,
    optimize_loss,
    DEFAULT_RCPS,
    initialize_params,
    logistic_ligand_map,
)
from .figures.common import getSetup


def plot_optimize(
    data: xarray.DataArray,
    opt_opts: Dict,
):
    """Run optimize_loss(), and show inferred vs actual Lbound"""
    rcps = opt_opts["rcps"]
    Ka = assemble_Ka(data, rcps).values
    x_opt, ctx = optimize_loss(data, **opt_opts)

    params = reshape_params(
        x_opt,
        data,
        opt_opts["logistic_ligands"],
        rcps=rcps,
    )

    Lbound = infer_Lbound(
        data,
        params["Rtot"],
        Ka,
        opt_opts["L0"],
        opt_opts["KxStar"],
        opt_opts["f"],
        opt_opts["logistic_ligands"],
        params["logistic_params"],
    )

    axes, fig = getSetup((8, 6), (1, 1))
    ax = plot_Lbound(data, Lbound, data.Ligand.values, ax=axes[0])

    return fig, Lbound, params


def plot_Lbound(
    data: xarray.DataArray,
    lbound: Union[xarray.DataArray, np.ndarray],
    lig: Union[Collection[str], str],
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
    if isinstance(lig, str):
        lig = [lig]
    idx = np.isin(data.Ligand.values, lig)
    data_filtered_flat = data.isel(Ligand=idx).values.swapaxes(0, 1).flatten()
    lbound_filtered_flat = lbound[:, idx].swapaxes(0, 1).flatten()
    assert lbound_filtered_flat.shape == data_filtered_flat.shape
    labels = np.concatenate(
        [np.full(data.sizes["Complex"], l) for l in data.Ligand.values[idx]]
    )
    ax = sns.scatterplot(
        x=np.log10(data_filtered_flat),
        y=np.log10(lbound_filtered_flat),
        hue=labels,
        alpha=0.7,
        ax=ax,
        palette=palette,
    )
    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Inferred", fontsize=12)
    ax.legend(title="Ligand", bbox_to_anchor=(1, 1), borderaxespad=0)
    return ax


def LLigO(
    data: xarray.DataArray, lig: Union[Collection[str], str], Ka=None, **opt_opts
) -> np.ndarray:
    """
    Args:
        data: prepared data in DataArray form
        rec: collection or individual receptor as a string

    Returns:
        Params
    """
    if isinstance(lig, str):
        lig = [lig]
    idx = ~np.isin(data.Ligand.values, lig)
    # recompute FcIdx with the remaining receptors
    data_no_lig = data.isel(Ligand=idx)
    opt_kwargs_sub = copy.deepcopy(opt_opts)
    opt_kwargs_sub["L0"] = opt_kwargs_sub["L0"][idx]
    opt_kwargs_sub["KxStar"] = opt_kwargs_sub["KxStar"][idx]
    opt_kwargs_sub["f"] = opt_kwargs_sub["f"][idx]
    opt_kwargs_sub["logistic_ligands"] = opt_kwargs_sub["logistic_ligands"][idx]
    opt_x, _ = optimize_loss(data_no_lig, **opt_kwargs_sub)
    rcps = opt_kwargs_sub.get("rcps", DEFAULT_RCPS)
    params = reshape_params(opt_x, data, opt_kwargs_sub["logistic_ligands"], rcps=rcps)
    Lbound = infer_Lbound(
        data_no_lig,
        params["Rtot"],
        Ka,
        opt_kwargs_sub["L0"],
        opt_kwargs_sub["KxStar"],
        opt_kwargs_sub["f"],
        opt_kwargs_sub["logistic_ligands"],
        params["logistic_params"],
    )
    return Lbound, params


def plot_LLigO(
    data: xarray.DataArray, lig: Union[Collection[str], str], Ka=None, **opt_opts
) -> matplotlib.axes.Axes:
    """
    Trains the model on data that excludes receptor(s) specified by rec. Plots
    the correlation between predicted and actual data as a scatter plot as 2 plots:
        1. Excluded receptors
        2. All

    Args:
        data: Prepared data as DataArray or np array
        rec: receptor(s) to leave out

    Returns:
        axes for plot
    """
    if isinstance(lig, str):
        lig = [lig]
    Lbound, params = LLigO(data, lig, Ka=Ka, **opt_opts)

    palette_list = sns.color_palette("bright", data.Ligand.values.shape[0])
    palette = {r: color for r, color in zip(data.Ligand.values, palette_list)}

    axes, plot = getSetup((16, 6), (1, 2))
    all_minus_lig = [l for l in data.Ligand.values if l not in lig]

    ax = plot_Lbound(
        data.sel(Ligand=all_minus_lig),
        Lbound,
        all_minus_lig,
        ax=axes[0],
        palette=palette,
    )
    ax.set_title(f"All - {', '.join(lig)}", fontsize=15)

    if np.any(
        (data.Ligand.values == lig[0])
        & logistic_ligand_map(opt_opts["logistic_ligands"])
    ):
        # if 1 left out ligand is a logistic ligand, all left out ligands should be logistic ligands
        assert np.all(
            np.isin(data.Ligand.values, np.array(lig))
            == logistic_ligand_map(opt_opts["logistic_ligands"])
        )

        lig_idx = np.isin(data.Ligand.values, lig)
        rcp_idx = np.zeros((len(lig), len(opt_opts["rcps"])), dtype=bool)

        for io, l in enumerate(data.Ligand.values):
            if not lig_idx[io]:
                continue
            ii = np.where(np.array(lig) == l)[0]
            if np.any(opt_opts["logistic_ligands"][io]):
                rcp_idx[ii] = opt_opts["logistic_ligands"][io]
            else:
                rcp_idx[ii] = np.isin(np.array(opt_opts["rcps"]), lig)
        inferred = params["Rtot"] @ rcp_idx.T
    else:
        Ka = assemble_Ka(data, opt_opts["rcps"]).values
        lig_idx = np.isin(data.Ligand.values, lig)
        inferred = infer_Lbound_mv(
            params["Rtot"],
            Ka[lig_idx],
            opt_opts["L0"][lig_idx],
            opt_opts["KxStar"][lig_idx],
            opt_opts["f"][lig_idx],
        )

    ax = sns.scatterplot(
        x=np.log10(data.sel(Ligand=lig).values.swapaxes(1, 0)).flatten(),
        y=np.log10(inferred.swapaxes(1, 0)).flatten(),
        hue=np.concatenate([np.full(data.shape[0], l) for l in lig]),
        ax=axes[1],
    )
    ax.set_title(f"{', '.join(lig)}", fontsize=15)
    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Inferred", fontsize=12)

    return plot, Lbound, params
