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
    Lbound = infer_Lbound(data, *(params + [Ka]), L0, KxStar, f)
    res = (np.nan_to_num(np.log(data) - np.log(Lbound), neginf=0, posinf=0)).flatten()
    return res


def infer_Lbound(
    cube: np.ndarray,
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
      cube: Raw data of shape (n_samp, n_rcp, n_ag)
      Rtot: Antibody abundances of shape (n_samp * n_ag, n_ab)
      Ka: Affinities of shape (n_rcp, n_ab)
      L0: 1D array of size n_rcp
      KxStar: 1D array of size n_rcp
      f: 1D array of size n_rcp

    Returns:
      The amount of each ligand bound for each sample and antigen. Same size as
      cube.
    """
    Rtot = Rtot.reshape((cube.shape[0], Rtot.shape[1], cube.shape[2]))
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

    arrgs = (
        data.values,
        Ka,
        L0,
        KxStar,
        f,
        ab_types,
        fitKa,
    )

    # Setup a matrix characterizing the block sparsity of the Jacobian
    rcp_ab_block = np.ones((n_rcp, n_ab), dtype=int)
    jac_sparsity = np.zeros((n_samp, n_ag, n_samp, n_ag, n_rcp, n_ab), dtype=int)
    samp_ag = np.zeros((n_samp, n_ag), dtype=int)
    idx = np.indices(samp_ag.shape)
    jac_sparsity[*idx, *idx] = rcp_ab_block
    jac_sparsity = np.moveaxis(jac_sparsity, (4, 5), (1, 4))
    n_out = n_samp * n_rcp * n_ag
    n_in = n_samp * n_ab * n_ag
    jac_sparsity = jac_sparsity.reshape(n_out, n_in)
    if fitKa:
        Ka_len = np.size(params[-1])
        jac_sparsity = np.hstack((jac_sparsity, np.ones((n_out, Ka_len))))
    jac_sparsity = coo_matrix(jac_sparsity)

    print("")
    opt = least_squares(
        model_loss,
        log_x0,
        args=arrgs,
        verbose=2,
        jac_sparsity=jac_sparsity,
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
    abundance = x[:abundance_len].reshape((n_subj * n_ag, n_ab))
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
