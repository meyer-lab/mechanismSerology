from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from sklearn.metrics import r2_score
from statsmodels.multivariate.pca import PCA

from maserol.core import infer_Lbound, optimize_loss
from maserol.util import assemble_Ka, assemble_options


def assemble_residual_mask(data: xr.DataArray, ligand_missingness: Dict):
    """
    Adds missingness along particular ligands specified in `ligand_missingness`.
    """
    mask = data.copy()
    mask.values = np.ones_like(data, dtype=bool)
    n_cplx = data.shape[0]
    for ligand, missingness in ligand_missingness.items():
        if not isinstance(ligand, tuple):
            ligand = (ligand,)
        n_dropped = int(n_cplx * missingness)
        zeros = np.random.choice(n_cplx, n_dropped, replace=False)
        for lig in ligand:
            mask.sel(Ligand=lig)[zeros] = 0
    return mask.values


def impute_missing_ms(tensor, residual_mask, opts=None):
    """
    Imputes the values corresponding to the indices for which residual_mask ==j
    False using mechanistic serology.
    """
    assert residual_mask.dtype == bool
    default_opts = assemble_options(tensor)
    default_opts["tol"] = 1e-5
    if opts is not None:
        default_opts.update(opts)
    opts = default_opts
    opts["residual_mask"] = residual_mask
    params, ctx = optimize_loss(tensor, **opts, return_reshaped_params=True)
    # subset the options
    Lbound_opt_names = ["L0", "KxStar", "f", "logistic_ligands"]
    Lbound_opts = {opt_name: opts[opt_name] for opt_name in Lbound_opt_names}
    Lbound = infer_Lbound(
        tensor,
        params["Rtot"],
        Ka=assemble_Ka(
            tensor.Ligand.values,
            rcps=opts["rcps"],
            logistic_ligands=opts["logistic_ligands"],
        ).values,
        logistic_params=params["logistic_params"],
        **Lbound_opts,
    )
    return Lbound


def impute_missing_pca(tensor, residual_mask, ncomp=5):
    """
    Imputes the values corresponding to the indices for which residual_mask ==j
    False using PCA.
    """
    assert residual_mask.dtype == bool
    tensor = np.log(tensor + 1)
    tensor.values[~residual_mask] = np.NaN
    opt = PCA(tensor, ncomp, missing="fill-em")
    return np.exp(opt.projection) - 1


def imputation_scatterplot(tensor, Lbound, residual_mask, ax):
    """Plots imputed vs actual."""
    assert residual_mask.dtype == bool
    assert tensor.shape == Lbound.shape
    test_mask = ~residual_mask
    y = np.log10(tensor.values[test_mask] + 1)
    x = np.log10(Lbound[test_mask] + 1)

    df = pd.DataFrame(
        {
            "Imputed": x,
            "Actual": y,
        }
    )
    _, lig_idx = np.where(test_mask)
    multi_lig = len(set(lig_idx)) > 1
    if multi_lig:
        df["Ligand"] = np.array([tensor.Ligand.values[i] for i in lig_idx])
    ax = sns.scatterplot(
        data=df,
        x="Imputed",
        y="Actual",
        hue="Ligand" if multi_lig else None,
        ax=ax,
        alpha=0.72,
    )
    r = np.corrcoef(x, y)[0][1]
    ax.annotate(f"$r$ = {r:.2f}", xy=(0.7, 0.1), xycoords="axes fraction")
    # also show the R2
    r2 = r2_score(y, x)
    # use latex for the R2
    ax.annotate(f"$R^2$ = {r2:.2f}", xy=(0.7, 0.04), xycoords="axes fraction")
    return ax


def run_repeated_imputation(
    tensor, imputer, ligs=None, missingness=0.1, runs=5, imputer_name=None
):
    """
    Run imputation one or more times for all of the ligands in `ligs`. Returns
    pd.DataFrame with results from each run.
    """
    df = pd.DataFrame(columns=["Method", "r2", "r", "Ligand", "Missingness"])
    for lig in ligs or tensor.Ligand.values:
        for _ in range(runs):
            residual_mask = assemble_residual_mask(tensor, {lig: missingness})
            actual = np.log10(tensor.values[~residual_mask] + 1)
            Lbound = np.log10(imputer(tensor, residual_mask)[~residual_mask] + 1)
            df.loc[len(df)] = [
                # name of function (accessible through func if
                # functools.partial)
                imputer_name
                or (
                    imputer.__name__
                    if hasattr(imputer, "__name__")
                    else imputer.func.__name__
                ),
                r2_score(actual, Lbound),
                np.corrcoef(actual, Lbound)[0, 1],
                lig,
                missingness,
            ]
    return df
