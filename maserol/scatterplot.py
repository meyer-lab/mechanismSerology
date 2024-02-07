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
    logistic_ligand_map,
)
from .figures.common import getSetup
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def plot_optimize(
    data: xarray.DataArray,
    opt_opts: Dict,
):
    """Run optimize_loss(), and show inferred vs actual Lbound"""
    rcps = opt_opts["rcps"]
    Ka = assemble_Ka(
        data.Ligand.values, rcps, logistic_ligands=opt_opts["logistic_ligands"]
    ).values
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
    ax = plot_Lbound(
        data, Lbound, data.Ligand.values, ax=axes[0], with_r2=True, with_fit_line=True
    )

    return fig, Lbound, params


def plot_Lbound(
    data: xarray.DataArray,
    Lbound: Union[xarray.DataArray, np.ndarray],
    lig: Union[Collection[str], str],
    ax=None,
    palette: Optional[Union[List, Mapping]] = None,
    with_r2=False,
    with_fit_line=False,
) -> matplotlib.axes.Axes:
    """
    Plots the lbound predictions vs their actual values on a scatter plot.

    Args:
        data: Prepared data as DataArray
        lbound: Predicted lbound as numpy array or DataArray
        lig: Ligands to plot

    Returns:
      matplotlib axis for the created plot
    """
    if isinstance(lig, str):
        lig = [lig]
    idx = np.isin(data.Ligand.values, lig)
    data_filtered_flat = data.isel(Ligand=idx).values.swapaxes(0, 1).flatten()
    Lbound_filtered_flat = Lbound[:, idx].swapaxes(0, 1).flatten()
    assert Lbound_filtered_flat.shape == data_filtered_flat.shape
    labels = np.concatenate(
        [np.full(data.sizes["Complex"], l) for l in data.Ligand.values[idx]]
    )
    actual = np.log10(data_filtered_flat)
    inferred = np.log10(Lbound_filtered_flat)
    ax = sns.scatterplot(
        x=actual,
        y=inferred,
        hue=labels,
        alpha=0.7,
        ax=ax,
        palette=palette,
    )
    if with_r2:
        add_r2_label(ax, actual, inferred)
    if with_fit_line:
        add_slope_1_line(ax, actual, inferred)
    ax.set_xlabel("Actual", fontsize=9)
    ax.set_ylabel("Inferred", fontsize=9)
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
    mv_idx = ~np.isin(
        data.Ligand.values[~logistic_ligand_map(opt_opts["logistic_ligands"])], lig
    )
    idx = ~np.isin(data.Ligand.values, lig)
    # recompute FcIdx with the remaining receptors
    data_no_lig = data.isel(Ligand=idx)
    opt_kwargs_sub = copy.deepcopy(opt_opts)
    opt_kwargs_sub["L0"] = opt_kwargs_sub["L0"][mv_idx]
    opt_kwargs_sub["KxStar"] = opt_kwargs_sub["KxStar"][mv_idx]
    opt_kwargs_sub["f"] = opt_kwargs_sub["f"][mv_idx]
    opt_kwargs_sub["logistic_ligands"] = opt_kwargs_sub["logistic_ligands"][idx]
    opt_x, _ = optimize_loss(data_no_lig, **opt_kwargs_sub)
    rcps = opt_kwargs_sub.get("rcps", DEFAULT_RCPS)
    params = reshape_params(opt_x, data, opt_kwargs_sub["logistic_ligands"], rcps=rcps)
    Lbound = infer_Lbound(
        data_no_lig,
        params["Rtot"],
        (
            Ka
            if Ka is not None
            else assemble_Ka(
                data_no_lig.Ligand.values, rcps, opt_kwargs_sub["logistic_ligands"]
            ).values
        ),
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

    axes, plot = getSetup((8, 3.5), (1, 2))
    all_minus_lig = [l for l in data.Ligand.values if l not in lig]

    ax = plot_Lbound(
        data.sel(Ligand=all_minus_lig),
        Lbound,
        all_minus_lig,
        ax=axes[0],
        palette=palette,
    )
    ax.set_title(f"Fitting all except {', '.join(lig)}", fontsize=12)

    lig_idx = np.isin(data.Ligand.values, lig)
    lig_idx_mv = np.isin(
        data.Ligand.values[~logistic_ligand_map(opt_opts["logistic_ligands"])], lig
    )
    if np.any(
        (data.Ligand.values == lig[0])
        & logistic_ligand_map(opt_opts["logistic_ligands"])
    ):
        # if 1 left out ligand is a logistic ligand, all left out ligands should be logistic ligands
        assert np.all(
            np.isin(data.Ligand.values, np.array(lig))
            == logistic_ligand_map(opt_opts["logistic_ligands"])
        )

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
        Ka = assemble_Ka(data.Ligand.values[lig_idx], opt_opts["rcps"]).values
        inferred = infer_Lbound_mv(
            params["Rtot"],
            Ka,
            opt_opts["L0"][lig_idx_mv],
            opt_opts["KxStar"][lig_idx_mv],
            opt_opts["f"][lig_idx_mv],
        )

    actual = np.log10(data.sel(Ligand=lig).values.swapaxes(1, 0)).flatten()
    inferred = np.log10(inferred.swapaxes(1, 0)).flatten()
    ax = sns.scatterplot(
        x=actual,
        y=inferred,
        hue=np.concatenate([np.full(data.shape[0], l) for l in lig]),
        ax=axes[1],
    )
    add_r2_label(axes[1], actual, inferred)
    add_slope_1_line(axes[1], actual, inferred)
    ax.set_title(f"Predicted binding of {', '.join(lig)}", fontsize=12)
    ax.set_xlabel("Actual", fontsize=9)
    ax.set_ylabel("Inferred", fontsize=9)

    return plot, Lbound, params


def add_r2_label(ax, actual, inferred):
    r2 = r2_score(actual, inferred)
    pearson = pearsonr(actual, inferred)[0]
    ax.text(
        0.95,
        0.05,
        r"$\mathit{r}$=" + str(round(pearson, 2)),
        verticalalignment="bottom",
        horizontalalignment="right",
        size=15,
        transform=ax.transAxes,
    )


def add_slope_1_line(ax, actual, inferred):
    low = min(np.min(actual), np.min(inferred))
    high = max(np.max(actual), np.max(inferred))
    ax.plot([low, high], [low, high], "k--", alpha=0.5)
