from typing import Collection, Mapping, Optional

from .preprocess import makeRcpAgLabels, HIgGs
from .core import *
from .figures.common import *

def plotPrediction(data: xr.DataArray, lbound, ax=None, logscale=True):
    """ Create a basic figure of actual vs predict scatterplot. """
    assert data.shape == lbound.shape
    cube_flat = data.values.flatten()

    valid = getNonnegIdx(data)
    receptor_labels, antigen_labels = makeRcpAgLabels(data)

    x = cube_flat[valid]
    y = lbound.flatten()[valid]
    if logscale:
        x = np.log(x)
        y = np.log(y)

    # plot
    f = sns.scatterplot(x=x,
                        y=y, 
                        hue=receptor_labels[valid],
                        style=antigen_labels[valid],
                        ax=ax, s=70, alpha=0.5)
    f.legend(title="Receptor | Antigen", bbox_to_anchor=(1, 1), borderaxespad=0)
    f.set_xlabel("Actual", fontsize=12)
    f.set_ylabel("Predictions", fontsize=12)
    return f

def plotOptimize(data: xr.DataArray, metric="mean", lrank=True, fitKa=False,
                 ab_types=HIgGs, maxiter=500):
    """ Run optimizeLoss(), and compare scatterplot before and after """
    cube = prepare_data(data)
    x_opt, f_opt, init_p = optimizeLoss(cube, metric=metric, lrank=lrank, fitKa=fitKa,
                                        ab_types=ab_types, maxiter=maxiter, retInit=True)

    init_lbound = inferLbound(cube, *init_p, lrank=lrank)

    new_p = reshapeParams(x_opt, cube, lrank=lrank, fitKa=fitKa, ab_types=ab_types)
    if not fitKa:
        new_p.append(init_p[-1])
    final_lbound = inferLbound(cube, *new_p, lrank=lrank)

    if metric.startswith("mean"):
        if metric.endswith("autoscale"):
            if metric.startswith("mean_rcp"):
                log_diffs = np.nan_to_num(np.log(cube.values) - np.log(final_lbound))
                final_scaler = np.mean(np.mean(log_diffs, axis=2), axis=0)[:, np.newaxis]
            else:
                log_diffs = np.nan_to_num(np.log(cube.values) - np.log(final_lbound))
                final_scaler = np.mean(log_diffs)
            init_scaler = 1 # we want to see the true initial Lbound distribution
        elif metric.startswith("mean_rcp"):
            init_scaler = (np.random.rand(data.Receptor.size) * 2)[:, np.newaxis]
            final_scaler = x_opt[x_opt.size - cube.shape[1]:, np.newaxis]
        elif metric == "mean_direct":
            init_scaler = 0
            final_scaler = 0
        else:
            init_scaler = np.random.rand(1) * 2
            final_scaler = x_opt[-1]
        print("Init Scaling factors", init_scaler)
        print("Scaling factors:", final_scaler)
        init_lbound = np.log(init_lbound) + init_scaler
        final_lbound = np.log(final_lbound) + final_scaler
        if not metric.endswith("no_log"):
            cube = np.log(cube)

    axs, f = getSetup((13, 5), (1, 2))
    sns.set(style="darkgrid", font_scale=1)
    initial_f = plotPrediction(cube, init_lbound, axs[0], logscale=not metric.startswith("mean"))
    initial_f.set_title("Initial", fontsize=13)
    new_f = plotPrediction(cube, final_lbound, axs[1], logscale=not metric.startswith("mean"))
    new_f.set_title("After Abundance Fit", fontsize=13)

    # Add R numbers onto plot
    Raxis = -1
    if metric == "rrcp":
        Raxis = 1
    if metric == "rag":
        Raxis = 2
    f.text(0.05, 0.1, gen_R_labels(cube, init_lbound, Raxis), fontsize=12)
    f.text(0.55, 0.1, gen_R_labels(cube, final_lbound, Raxis), fontsize=12)
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


def plotLbound(data: xr.DataArray, lbound: Union[xr.DataArray, np.ndarray],
               rec: Union[Collection[str], str], ax=None,
               palette: Optional[Union[List, Mapping]] = None) -> matplotlib.axes.Axes:
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
        [np.ravel(t) for t in np.split(data_filtered.values, data_filtered.values.shape[1], axis=1)])
    lbound_filtered = lbound[:, idx[0], :]
    lbound_filtered_flat = np.concatenate(
        [np.ravel(t) for t in np.split(lbound_filtered, lbound_filtered.shape[1], axis=1)])
    assert lbound_filtered_flat.shape == data_filtered_flat.shape
    n_points_per_rec = lbound_filtered_flat.shape[0] / len(rec)
    assert n_points_per_rec.is_integer()
    n_points_per_rec = int(n_points_per_rec)
    labels = np.concatenate([np.full(n_points_per_rec, r) for r in data.Receptor.values[idx[0]]])
    f = sns.scatterplot(x=np.log(data_filtered_flat), y=np.log(lbound_filtered_flat), hue=labels, alpha=0.8, ax=ax,
                        palette=palette)
    f.set_xlabel("Actual", fontsize=12)
    f.set_ylabel("Predictions", fontsize=12)
    f.legend(title="Receptor", bbox_to_anchor=(1, 1), borderaxespad=0)
    return f


def LRcpO(data: xr.DataArray, rec: Union[Collection[str], str], **opt_kwargs) -> np.ndarray:
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


def plotLRcpO(data: Union[xr.DataArray, np.ndarray], rec: Union[Collection[str], str],
               **opt_kwargs) -> matplotlib.axes.Axes:
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




