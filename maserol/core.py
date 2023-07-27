"""
Core function for serology mechanistic tensor factorization
"""
# Base Python
from typing import Collection, Dict, List, Optional, Tuple, Union

# Extended Python
import numpy as np
import xarray as xr
from scipy.optimize import least_squares, newton
from scipy.sparse import coo_matrix

# Current Package
from .preprocess import assemble_Ka, HIgGs, DEFAULT_AB_TYPES


"""
This implementation relies on the multivalent binding model paper:
https://www.sciencedirect.com/science/article/pii/S002555642100122X. 
"""


def model_loss(
    log_x: np.ndarray,
    data: Union[xr.DataArray, np.ndarray],
    Ka: np.ndarray,
    L0: np.ndarray,
    KxStar: np.ndarray,
    f: np.ndarray,
    intersample_detections: np.ndarray[bool],
    ab_types: tuple[str],
    fitKa: Optional[Union[tuple, bool]] = False,
) -> np.ndarray:
    """
    Computes the loss, comparing model output and actual measurements

    Args:
      log_x: 1D array with model parameters
      data: Raw data of shape (n_samp, n_rcp, n_ag)
      Ka: Affinities of shape (n_rcp, n_ab)
      L0: 1D array of size n_rcp
      KxStar: 1D array of size n_rcp
      f: 1D array of size n_rcp
      intersample_detections: boolean mask indicating which detection reagents
        should be treated as intersample information.
      ab_types: tuple containing the antibody types (length n_ab)

    Returns:
      The loss
    """
    params = reshape_params(log_x, data, ab_types=ab_types, fitKa=fitKa)
    if fitKa:
        if isinstance(fitKa, tuple):
            Ka = np.copy(Ka)
            Ka[fitKa] = params[-1]
        else:
            Ka = params[-1]
        params = params[:-1]
    if isinstance(data, xr.DataArray):
        data = np.array(data)
    Lbound = infer_Lbound(*(params + [Ka]), L0, KxStar, f)

    intra_loss = (
        np.nan_to_num(
            np.log(data[:, ~intersample_detections, :])
            - np.log(Lbound[:, ~intersample_detections, :]),
            neginf=0,
            posinf=0,
        )
    ).flatten()
    inter_loss = intersample_loss(
        data[:, intersample_detections, :], Lbound[:, intersample_detections, :]
    ).flatten()

    inter_loss = inter_loss * data.shape[1] / np.sum(intersample_detections) * 3
    return np.concatenate((intra_loss, inter_loss))


def infer_Lbound(
    Rtot: np.ndarray,
    Ka: np.ndarray,
    L0: np.ndarray,
    KxStar: np.ndarray,
    f: np.ndarray,
):
    """
    Infers the amount of bound ligand (detection reagent) from the antibody
    abundances, Rtot. See (4.14) in binding model paper.

    Args:
      Rtot: Antibody abundances of shape (n_samp, n_ab, n_ag)
      Ka: Affinities of shape (n_rcp, n_ab)
      L0: 1D array of size n_rcp
      KxStar: 1D array of size n_rcp
      f: 1D array of size n_rcp

    Returns:
      The amount of each ligand bound for each sample and antigen. Same size as
      cube.
    """
    Phi = custom_root(Rtot, L0, KxStar, Ka, f)
    return (
        L0[:, np.newaxis]
        / KxStar[:, np.newaxis]
        * ((1.0 + Phi) ** f[:, np.newaxis] - 1.0)
    )


def phi(
    Phi: np.ndarray, KaRT: np.ndarray, fLKa: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """
    Updates the value of Phi.

    Args:
      Phi: Existing value of Phi of shape (n_samp, n_rcp, n_ag)
      KaRT: Product of Ka, Rtot, KxStar of shape (n_samp, n_rcp, n_ab, n_ag)
      fLKa: Product of f, L0, Ka of shape (1, n_rcp, n_ab, 1)
      f: 1D array of valencies of size n_rcp

    Returns:
      Phi of shape (n_samp, n_rcp, n_ag)
    """
    # n_samp * n_rcp * n_ag
    Phi_temp = np.sum(
        KaRT
        / (
            1.0
            + fLKa
            * (1.0 + Phi[:, :, np.newaxis, :]) ** (f[:, np.newaxis, np.newaxis] - 1)
        ),
        axis=2,
    )
    assert Phi_temp.shape == Phi.shape
    return Phi_temp


def custom_root(
    Rtot: np.ndarray, L0: np.ndarray, KxStar: np.ndarray, Ka: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """
    Computes Phi from Rtot using mass balance and numerical optimization.

    Args:
      Rtot: Antibody abundances of shape (n_samp, n_ab, n_ag)
      L0: 1D array of size n_rcp
      KxStar: 1D array of size n_rcp
      Ka: Matrix of shape (n_rcp, n_ab)
      f: 1D array of size n_rcp

    Returns:
      Phi of shape (n_samp, n_rcp, n_ag)
    """
    # Precalculate these quantities for speed
    # n_samp * n_rcp * n_ab * n_ag
    KaRT = (
        Ka[np.newaxis, :, :, np.newaxis]
        * Rtot[:, np.newaxis, :, :]
        * KxStar[:, np.newaxis, np.newaxis]
    )
    # 1 * n_rcp * n_ab * 1
    fLKa = (
        f[:, np.newaxis, np.newaxis]
        * L0[:, np.newaxis, np.newaxis]
        * Ka[np.newaxis, :, :, np.newaxis]
    )

    # Solve for an initial guess by using f = 2
    a = np.sum(fLKa, axis=2)
    b = 1 + a
    # quadratic formula
    f0 = (-b + np.sqrt(b**2 + 4 * a * np.sum(KaRT, axis=2))) / 2 / a

    root = newton(
        lambda *args: phi(*args) - args[0], f0, maxiter=15000, args=(KaRT, fLKa, f)
    )

    assert np.all(root >= 0)
    return root


def optimize_loss(
    data: xr.DataArray,
    L0: np.ndarray,
    KxStar: np.ndarray,
    f: np.ndarray,
    ab_types: Optional[tuple[str]] = HIgGs,
    fitKa: Optional[Union[tuple, bool]] = False,
    params: Optional[List] = None,
    intersample_detections: Optional[np.ndarray[bool]] = None,
    ftol: float = 1e-7,
    gtol: float = 1e-7,
    xtol: float = 1e-7,
) -> Tuple[np.ndarray, Dict]:
    """
    Infers the antibody abundances given the systems serology data in the data
    parameter.

    Args:
      data: DataArray with systems serology measurements. Shape: (n_samp, n_rcp, n_ag)
      L0: 1D array of size n_rcp. Concentration of each ligand.
      KxStar: 1D array of size n_rcp. KxStar (for detailed balance correction) of each ligand.
      f: 1D array of size n_rcp. valency of each ligand.
      ab_types: ab types to fit the data
      fitKa: Either a bool (False: treat Ka as fixed, True: optimize Ka) or a
        tuple of indices into the Ka array which specify specific affinities to
        fit.
      intersample_detections: boolean mask indicating which detection reagents
        should be treated as intersample information.
      params: starting parameters (Rtot, Ka) to use
      ftol: optimization tolerance: see scipy.optimize.least_squares
      gtol: optimization tolerance: see scipy.optimize.least_squares
      xtol: optimization tolerance: see scipy.optimize.least_squares

    Returns:
      (x, ctx) where x are the parameters achieved in the optimization (see
      reshape_params), and ctx is a dictionary containing additional context
      about the optimization.
    """
    n_samp, n_rcp, n_ag = data.shape
    n_ab = len(ab_types)
    assert L0.shape == (n_rcp,), "L0 wrong shape"
    assert KxStar.shape == (n_rcp,), "KxStar wrong shape"
    assert f.shape == (n_rcp,), "f wrong shape"

    if params is None:
        params = initialize_params(data, ab_types=ab_types)
    Ka = params[-1]
    if fitKa:
        if isinstance(fitKa, tuple):
            # fit specific elements of Ka
            params[-1] = params[-1][fitKa]
        # if fitKa is True, fit the entire Ka matrix
    else:
        params = params[:-1]

    log_x0 = flatten_params(*params)

    intersample_detections = (
        intersample_detections
        if intersample_detections is not None
        else np.zeros(n_rcp, dtype=bool)
    )

    arrgs = (data.values, Ka, L0, KxStar, f, intersample_detections, ab_types, fitKa)

    print("")
    opt = least_squares(
        model_loss,
        log_x0,
        args=arrgs,
        verbose=2,
        jac_sparsity=assemble_jac_sparsity(
            data, ab_types, intersample_detections, params, fitKa
        ),
        ftol=ftol,
        gtol=gtol,
        xtol=xtol,
    )

    if not fitKa:
        params.append(Ka)
    ctx = {"opt": opt, "init_params": params}
    ret = (opt.x, ctx)
    return ret


def reshape_params(
    log_x: np.ndarray,
    data,
    fitKa: Optional[Union[tuple, bool]] = False,
    ab_types: Collection = DEFAULT_AB_TYPES,
    as_xarray: bool = False,
):
    """Reshapes vector, x, into matrices. Inverse operation of flatten_params()."""
    x = np.exp(log_x)
    n_subj, n_rec, n_ag = data.shape
    n_ab = len(ab_types)

    non_ka_params_len = abundance_len = n_subj * n_ag * n_ab
    abundance = x[:abundance_len].reshape((n_subj, n_ab, n_ag))
    if as_xarray:
        assert isinstance(
            data, xr.DataArray
        ), "data parameter must be an xarray.DataArray"
        abundance = abundance.reshape((n_subj, n_ab, n_ag))
        abundance = xr.DataArray(
            abundance,
            (data.Sample.values, list(ab_types), data.Antigen.values),
            ("Sample", "Antibody", "Antigen"),
        )
    retVal = [abundance]

    if fitKa:  # retrieve Ka from x as well
        if isinstance(fitKa, tuple):
            # return fitted affinities as 1d array
            ka_len = len(fitKa[0])
            Ka = x[non_ka_params_len : non_ka_params_len + ka_len]
        else:
            # return entire Ka matrix
            ka_len = n_rec * n_ab
            Ka = x[non_ka_params_len : non_ka_params_len + ka_len].reshape(n_rec, n_ab)
            if as_xarray:
                Ka = xr.DataArray(
                    Ka, (data.Receptor.values, list(ab_types)), ("Receptor", "Antibody")
                )
        retVal.append(Ka)
    return retVal


def flatten_params(*args):
    """Flatten into a parameter vector. Inverse operation of reshape_params().
    Order: (r_subj, r_ag) / abund, Ka"""
    return np.log(np.concatenate([a.flatten() for a in args]))


def initialize_params(
    data: xr.DataArray, ab_types: Collection = DEFAULT_AB_TYPES
) -> List:
    Ka = assemble_Ka(data, ab_types).values
    log_mean = np.mean(np.nan_to_num(np.log10(data.values + 1)))  # deal with 0s
    abundance = np.random.uniform(
        10 ** (log_mean - 4),
        10 ** (log_mean + 2),
        (data.sizes["Sample"], len(ab_types), data.sizes["Antigen"]),
    )
    return [abundance, Ka]


def intersample_loss(data: np.ndarray, Lbound: np.ndarray) -> np.ndarray:
    """
    Computes the loss based on the relative difference of detection reagents
    between samples in data.

    Args:
      data: Raw data
      Lbound: Inferred Lbound

    Returns:
      loss (in the same shape as Lbound)
    """
    n_samp, n_rcp, n_ag = data.shape
    cube_to_matrix = lambda x: np.moveaxis(x, 2, 1).reshape(n_samp * n_ag, n_rcp)
    matrix_to_cube = lambda x: np.moveaxis(x.reshape(n_samp, n_ag, n_rcp), 1, 2)
    data, Lbound = cube_to_matrix(data), cube_to_matrix(Lbound)
    ordered_idx = np.argsort(data, axis=0)
    Lbound_ordered = Lbound[ordered_idx, np.arange(n_rcp)]
    diffs = np.diff(np.log(Lbound_ordered), axis=0)
    diffs[diffs > 0] = 0
    loss = np.zeros_like(Lbound)
    loss[ordered_idx[:-1], np.arange(n_rcp)] = diffs
    loss = matrix_to_cube(loss)
    return loss


def assemble_jac_sparsity(
    data: xr.DataArray,
    ab_types: Tuple,
    intersample_detections: np.ndarray[bool],
    params: List,
    fitKa: Union[tuple, bool],
):
    """
    Create Jacobian sparsity matrix for optimization.

    Args:
      data: Raw data
      ab_types: Antibody types
      intersample_detections: boolean mask indicating which detection reagents
        should be treated as intersample information.
      params: parameters used in optimization (before flattening)
      fitKa: fitKa option (see optimize_loss)

    Returns: Jacobian sparsity matrix.
    """
    n_samp, n_rcp, n_ag = data.shape
    n_ab = len(ab_types)

    n_inter_rcp = np.sum(intersample_detections)
    n_intra_rcp = n_rcp - n_inter_rcp

    # intrasample jac sparsity
    rcp_ab_block = np.ones((n_intra_rcp, n_ab), dtype=int)
    jac_sparsity_intra = np.zeros(
        (n_samp * n_ag, n_samp * n_ag, n_intra_rcp, n_ab), dtype=int
    )
    jac_sparsity_intra[
        np.arange(n_samp * n_ag), np.arange(n_samp * n_ag)
    ] = rcp_ab_block
    jac_sparsity_intra = jac_sparsity_intra.reshape(
        n_samp, n_ag, n_samp, n_ag, n_intra_rcp, n_ab
    )
    jac_sparsity_intra = np.moveaxis(jac_sparsity_intra, (4, 5), (1, 4))
    jac_sparsity_intra = jac_sparsity_intra.reshape(
        n_samp * n_intra_rcp * n_ag, n_samp * n_ab * n_ag
    )

    # intersample jac sparsity
    rcp_ab_block = np.ones((n_inter_rcp, n_ab), dtype=int)
    jac_sparsity_inter = np.zeros(
        (n_samp * n_ag, n_samp * n_ag, n_inter_rcp, n_ab), dtype=int
    )
    jac_sparsity_inter[
        np.arange(n_samp * n_ag), np.arange(n_samp * n_ag)
    ] = rcp_ab_block
    data_mat = np.moveaxis(data.values[:, intersample_detections, :], 2, 1).reshape(
        n_samp * n_ag, n_inter_rcp
    )
    ordered_idx = np.argsort(data_mat, axis=0)

    # Every pair of adjacent entries in ordered_idx should depend on each
    # other
    for sa_idx in range(n_samp * n_ag - 1):
        for rcp_idx in range(n_inter_rcp):
            jac_sparsity_inter[
                ordered_idx[sa_idx], ordered_idx[sa_idx + 1], rcp_idx
            ] = np.ones(n_ab, dtype=int)
            jac_sparsity_inter[
                ordered_idx[sa_idx + 1], ordered_idx[sa_idx], rcp_idx
            ] = np.ones(n_ab, dtype=int)
    jac_sparsity_inter = jac_sparsity_inter.reshape(
        (n_samp, n_ag, n_samp, n_ag, n_inter_rcp, n_ab)
    )
    jac_sparsity_inter = np.moveaxis(jac_sparsity_inter, (4, 5), (1, 4))
    jac_sparsity_inter = jac_sparsity_inter.reshape(
        n_samp * n_inter_rcp * n_ag, n_samp * n_ab * n_ag
    )

    jac_sparsity = np.vstack((jac_sparsity_intra, jac_sparsity_inter))

    if fitKa:
        Ka_len = np.size(params[-1])
        jac_sparsity = np.hstack(
            (jac_sparsity, np.ones((n_samp * n_rcp * n_ag, Ka_len)))
        )

    return coo_matrix(jac_sparsity)
