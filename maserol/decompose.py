import numpy as np
import xarray as xr
from sklearn.decomposition import NMF

def factorAbundance(abundance: xr.DataArray, n_comps: int, as_xarray=True):
    """
    Factors full-rank abundance tensor into two tensors.

    Args:
        abundance: abundance tensor as an xarray
        n_comps: number of components in factorization
        as_xarray: if true, return resulting factors in xarrays

    Returns:
        Two factor tensors.
        1. Sample factors, with shape (n_samples, n_abs, n_comps)
        2. Ag factors, with shape (n_ag, n_abs, n_comps)
    """
    assert isinstance(abundance, xr.DataArray), "Abundance must be passed as DataArray for factorization"
    n_abs = len(abundance.Antibody)
    sample_facs = np.zeros((len(abundance.Sample), n_abs, n_comps))
    ag_facs = np.zeros((len(abundance.Antigen), n_abs, n_comps))
    for ab_idx in range(n_abs):
        mat = abundance.isel(Antibody=ab_idx).values
        model = NMF(n_comps, max_iter=5_000)
        sample_slice = model.fit_transform(mat)
        ag_slice = model.components_
        # move the weight from ag_slice to sample_slice
        ag_weight = np.max(ag_slice)
        ag_slice = ag_slice / ag_weight
        sample_slice = sample_slice * ag_weight
        sample_facs[:, ab_idx, :] = sample_slice
        ag_facs[:, ab_idx, :] = ag_slice.T
    if as_xarray:
        # component names will be 1-indexed
        comp_names = np.arange(1, n_comps + 1)
        sample_facs = xr.DataArray(
            sample_facs,
            (abundance.Sample.values, abundance.Antibody.values, comp_names),
            ("Sample", "Antibody", "Component"),
        )
        ag_facs = xr.DataArray(
            ag_facs,
            (abundance.Antigen.values, abundance.Antibody.values, comp_names),
            ("Antigen", "Antibody", "Component"),
        )
    return sample_facs, ag_facs


def reconstructAbundance(sample_facs, ag_facs):
    """
    Reconstructs abundance from factors
    """
    if isinstance(sample_facs, xr.DataArray):
        sample_facs = sample_facs.values
    if isinstance(ag_facs, xr.DataArray):
        ag_facs = ag_facs.values
    return np.einsum("ijl,kjl->ijk", sample_facs, ag_facs)