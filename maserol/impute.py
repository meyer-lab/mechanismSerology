from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from sklearn.metrics import r2_score
from statsmodels.multivariate.pca import PCA
from tensorly.decomposition import parafac

from maserol.preprocess import assemble_options
from maserol.scatterplot import plot_optimize


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
    default_opts = assemble_options(tensor)
    default_opts["tol"] = 1e-5
    if opts is not None:
        default_opts.update(opts)
    opts = default_opts
    opts["residual_mask"] = residual_mask
    ax, Lbound, params = plot_optimize(tensor, opts)
    plt.clf()
    return Lbound


def impute_missing_cp(tensor, residual_mask, ncomp=5):
    """
    Imputes the values corresponding to the indices for which residual_mask ==j
    False using CP.
    """
    tensor = np.log(np.copy(tensor.values))
    tensor[~residual_mask] = 0  # no cheating!
    weights, factors = parafac(tensor, rank=ncomp, mask=residual_mask)
    assert np.all(weights == 1)
    assert factors[0].shape[1] == factors[1].shape[1] == factors[2].shape[1]
    Lbound = np.einsum("ij,kj,lj->ikl", *factors)
    return np.exp(Lbound)


def impute_missing_pca(tensor, residual_mask, ncomp=5):
    """
    Imputes the values corresponding to the indices for which residual_mask ==j
    False using PCA.
    """
    tensor = np.log(tensor)
    tensor.values[~residual_mask] = np.NaN
    opt = PCA(tensor, ncomp, missing="fill-em")
    return np.exp(opt.projection)


def imputation_scatterplot(tensor, Lbound, residual_mask, ax):
    """Plots imputed vs actual."""
    assert tensor.shape == Lbound.shape
    test_mask = ~residual_mask
    print(tensor.shape)
    y = np.log10(tensor.values[test_mask])
    x = np.log10(Lbound[test_mask])

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
        data=df, x="Imputed", y="Actual", hue="Ligand" if multi_lig else None, ax=ax
    )
    r = np.corrcoef(x, y)[0][1]
    ax.annotate(f"r = {r:.2f}", xy=(0.8, 0.1), xycoords="axes fraction")


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
            actual = np.log10(tensor.values[~residual_mask])
            Lbound = np.log10(imputer(tensor, residual_mask)[~residual_mask])
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
