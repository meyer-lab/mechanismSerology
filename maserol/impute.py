from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
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
    if len(data.dims) == 3:
        return assemble_residual_mask_3d(data, ligand_missingness)
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


def assemble_residual_mask_3d(data: xr.DataArray, ligand_missingness: Dict):
    """
    Adds missingness along particular ligands in 3d tensor.
    """
    assert data.dims == ("Sample", "Ligand", "Antigen")
    mask = data.copy()
    mask.values = np.ones_like(data, dtype=bool)
    for ligand, missingness in ligand_missingness.items():
        if not isinstance(ligand, tuple):
            ligand = (ligand,)
        n_dropped = int(data.shape[0] * data.shape[2] * missingness)
        zeros = np.random.choice(
            data.shape[0] * data.shape[2], n_dropped, replace=False
        )
        for lig in ligand:
            mask.sel(Ligand=lig).values[
                [idx // data.shape[1] for idx in zeros],
                [idx % data.shape[1] for idx in zeros],
            ] = 0
    return mask.values


def impute_missing_ms(tensor, residual_mask):
    """
    Imputes the values corresponding to the indices for which residual_mask ==j
    False using mechanistic serology.
    """
    opts = assemble_options(tensor)
    opts["residual_mask"] = residual_mask
    opts["tol"] = 1e-5
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
    y = np.log10(tensor.values[test_mask])
    x = np.log10(Lbound[test_mask])
    ax = sns.scatterplot(x=x, y=y, ax=ax)
    r = np.corrcoef(x, y)[0][1]
    ax.annotate(f"r = {r:.2f}", xy=(0.8, 0.1), xycoords="axes fraction")
