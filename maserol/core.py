from collections.abc import Collection
from copy import deepcopy

import numpy as np
import xarray as xr
from scipy.optimize import least_squares, newton

from .util import (
    DEFAULT_RCPS,
    HIgGs,
    assemble_Ka,
    logistic_ligand_map,
    n_logistic_ligands,
)

"""
This is an implementation of the binding model from
https://www.sciencedirect.com/science/article/pii/S002555642100122X. 
"""

PARAM_ORDER = ("Rtot", "logistic_params")


def model_loss(
    log_x: np.ndarray,
    data: xr.DataArray,
    Ka: np.ndarray,
    L0: np.ndarray,
    KxStar: np.ndarray,
    f: np.ndarray,
    logistic_ligands: np.ndarray[bool],
    residual_mask: np.ndarray[bool],
    rcps: tuple[str],
) -> np.ndarray:
    """
    Args:
      log_x: 1D array with model parameters.
      data, Ka, L0, KxStar, f, logistic_ligands, residual_mask,
        rcps: see parameters with the same name in `optimize_loss` docstring.

    Returns:
      The loss: a 1D array of nonnegative floats
    """
    assert isinstance(data, xr.DataArray)

    params = reshape_params(
        log_x,
        data,
        logistic_ligands,
        rcps=rcps,
    )

    Lbound = infer_Lbound(
        data,
        params["Rtot"],
        Ka,
        L0,
        KxStar,
        f,
        logistic_ligands,
        params["logistic_params"],
    )

    Lbound_residuals = (
        np.where(
            residual_mask,
            np.nan_to_num(
                (np.log(Lbound + 1) - np.log(data.values + 1)),
                neginf=0,
                posinf=0,
            ),
            0,
        )
    ).flatten()

    logistic_slope_regularization = (
        np.maximum(params["logistic_params"][3] - 1.0, 0).flatten() ** 2
        * params["Rtot"].shape[0]
        * data.shape[1]
        / n_logistic_ligands(logistic_ligands)
    )

    return np.concatenate((Lbound_residuals, logistic_slope_regularization))


def infer_Lbound(
    data: xr.DataArray,
    Rtot: np.ndarray,
    Ka: np.ndarray,
    L0: np.ndarray,
    KxStar: np.ndarray,
    f: np.ndarray,
    logistic_ligands: np.ndarray,
    logistic_params: np.ndarray,
):
    """
    Infer ligand abundances from receptor abundances.

    Args:
        data: see `optimize_loss` docstring.
        Rtot: array of shape (n_cplx, n_rcp) containing receptor abundances.
        Ka, L0, KxStar, f, logistic_ligands: see `optimize_loss` docstring.
        logistic_params: array of shape (4, n_logistic_ligands) containing
            parameters for logistic binding curves

    Returns:
        Lbound: array of shape (n_cplx, n_lig)
    """
    Lbound = np.zeros(data.shape)
    logist_lig_map = logistic_ligand_map(logistic_ligands)
    Lbound[:, ~logist_lig_map] = infer_Lbound_mv(
        Rtot,
        Ka,
        L0,
        KxStar,
        f,
    )
    Lbound[:, logist_lig_map] = infer_Lbound_logistic(
        Rtot, logistic_params, logistic_ligands
    )
    return Lbound


def infer_Lbound_mv(
    Rtot: np.ndarray,
    Ka: np.ndarray,
    L0: np.ndarray,
    KxStar: np.ndarray,
    f: np.ndarray,
):
    """
    Infer ligand abundances from receptor abundances using multivalent binding
    model. See (4.14) in binding model paper.

    Args:
        Rtot: see `infer_Lbound`.
        Ka: matrix of shape (n_lig, n_rcp). Where n_lig may be smaller than
          n_lig in `infer_Lbound`.
        L0, KxStar, f: subset of L0, KxStar, f as seen in `infer_Lbound`
          containing entries only for the ligands which are to be inferred by
          multivalent binding model.

    Returns:
        Lbound array of shape (n_cplx, n_lig).
    """
    Phi = find_root_scipy(Rtot, L0, KxStar, Ka, f)
    return L0 / KxStar * ((1.0 + Phi) ** f - 1.0)


def infer_Lbound_logistic(
    Rtot: np.ndarray,
    logistic_params: np.ndarray,
    logistic_ligands: np.ndarray,
):
    """
    Infer ligand abundances using 4-parameter logistic curves (i.e. hill curve).

    Args:
        Rtot, logistic_params, logistic_ligands: see `infer_Lbound`.

    Returns:
        Lbound array of shape (n_cplx, n_lig).
    """
    assert np.all(
        np.abs(logistic_params[3]) < 3e1
    ), "Exponent overflow in logistic function"

    Lbound = logistic_params[1] + (logistic_params[0] - logistic_params[1]) / (
        1
        + (
            (Rtot @ logistic_ligands[np.any(logistic_ligands != 0, axis=1)].T)
            / logistic_params[2]
        )
        ** logistic_params[3]
    )
    return Lbound


def find_root_scipy(
    Rtot: np.ndarray, L0: np.ndarray, KxStar: np.ndarray, Ka: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """
    Find Phi given Rtot by finding roots of mass balance equation (see binding
    model paper 4.12) using scipy implementation of Newton-Raphson method.

    Args:
        Rtot: see `infer_Lbound`.
        L0, KxStar, Ka, f: see `optimize_loss`.

    Returns:
        Phi: array of shape (n_cplx, n_lig).
    """

    # Precalculate these quantities for speed
    KaRT = Ka[np.newaxis, :, :] * Rtot[:, np.newaxis, :] * KxStar[:, np.newaxis]
    fLKa = f[:, np.newaxis] * L0[:, np.newaxis] * Ka[np.newaxis, :, :]

    # Solve for an initial guess by using f = 2
    a = np.sum(fLKa, axis=2)
    b = 1 + a

    # quadratic formula
    f0 = (-b + np.sqrt(b**2 + 4 * a * np.sum(KaRT, axis=2))) / 2 / a

    root = newton(
        lambda *args: phi(*args) - args[0], f0, maxiter=10000, args=(KaRT, fLKa, f)
    )

    root[root < 0] = 0
    return root


def phi(
    Phi: np.ndarray, KaRT: np.ndarray, fLKa: np.ndarray, f: np.ndarray
) -> np.ndarray:
    """
    Updates the value of Phi.

    Args:
      Phi: Existing value of Phi of shape (n_cplx, n_lig)
      KaRT: Product of Ka, Rtot, KxStar of shape (n_cplx, n_lig, n_lig)
      fLKa: Product of f, L0, Ka of shape (1, n_lig, n_lig, 1)
      f: see `optimize_loss`.

    Returns:
      Phi of shape (n_cplx, n_lig).
    """
    # n_cplx * n_rcp
    Phi_temp = np.sum(
        KaRT / (1.0 + fLKa * (1.0 + Phi[:, :, np.newaxis]) ** (f[:, np.newaxis] - 1)),
        axis=2,
    )
    assert Phi_temp.shape == Phi.shape
    return Phi_temp


def optimize_loss(
    data: xr.DataArray,
    L0: np.ndarray,
    KxStar: np.ndarray,
    f: np.ndarray,
    rcps: tuple[str] = HIgGs,
    logistic_ligands: np.ndarray[bool] = None,
    residual_mask: np.ndarray[bool] = None,
    tol: float = 1e-6,
    Ka: np.ndarray[float] = None,
    return_reshaped_params=False,
) -> tuple[np.ndarray, dict]:
    """
    Infer the receptor abundances given the ligand abundances specified in `data`.

    Args:
        data: DataArray with systems serology measurements. Shape: (n_cplx,
          n_lig), where n_cplx is the number of immune complexes and n_lig is the
          number of ligands.
        L0: array of size (n_lig,) specifying the concentrations of each ligand
          in solution.
        KxStar: array of size (n_lig,) specifying the difference between free
          and multivalent binding affinity.
        f: array of size (n_lig,) specifying the valency of each ligand.
        rcps: Receptor (i.e. antibody) types to infer the abundance of.
        params: Initial parameters to use. If not provided, these will be
          randomly initialized.
        logistic_ligands: matrix of size (n_lig, n_rcp). Each row corresponds to
          the coefficients in the weighted sum of the receptors in that row. The
          weighted sum is then passed as input to the logistic curve, the output
          of which is the ligand abundance.
        residual_mask: Array with shape like data specifying, where 1 indicates
          that the corresponding entry in data is used in the optimization, and 0
          means the value is neglected.
        tol: Optimization tolerance. Used as gtol, xtol, ftol in
          scipy.optimize.least_squares.

    Returns:
        (x, ctx) where x are the parameters resulting from the optimization
        (flattened and logged), and ctx is a dictionary containing additional
        context about the optimization.
    """
    n_cplx, n_lig = data.shape
    n_rcp = len(rcps)
    assert np.all(data.values >= 0), "data must be nonnegative"
    n_mv_ligs = n_lig - n_logistic_ligands(logistic_ligands)
    assert L0.shape == n_mv_ligs, "L0 wrong shape"
    assert KxStar.shape == n_mv_ligs, "KxStar wrong shape"
    assert f.shape == n_mv_ligs, "f wrong shape"

    logistic_ligands = (
        logistic_ligands
        if logistic_ligands is not None
        else np.zeros((n_lig * n_rcp), dtype=bool)
    )

    Ka = (
        Ka
        if Ka is not None
        else assemble_Ka(data.Ligand.values, rcps, logistic_ligands).values
    )
    assert Ka.shape[0] == n_mv_ligs, "Ka wrong shape"

    residual_mask = (
        residual_mask if residual_mask is not None else np.ones_like(data, dtype=bool)
    )

    arrgs = (
        data,
        Ka,
        L0,
        KxStar,
        f,
        logistic_ligands,
        residual_mask,
        rcps,
    )

    print("")
    opt = least_squares(
        model_loss,
        flatten_params(
            initialize_params(
                data,
                logistic_ligands,
                rcps=rcps,
            )
        ),
        args=arrgs,
        verbose=2,
        jac_sparsity=assemble_jac_sparsity(
            data,
            rcps,
            logistic_ligands,
        ),
        bounds=assemble_bounds(data, rcps, logistic_ligands),
        x_scale=assemble_x_scale(data, rcps, logistic_ligands),
        ftol=tol,
        gtol=tol,
        xtol=tol,
    )

    ctx = {"opt": opt}
    params = (
        reshape_params(opt.x, data, logistic_ligands, rcps)
        if return_reshaped_params
        else opt.x
    )
    ret = (params, ctx)
    return ret


def reshape_params(
    log_x: np.ndarray,
    data,
    logistic_ligands: np.ndarray[bool],
    rcps: Collection = DEFAULT_RCPS,
):
    """
    Reshapes and scales up x.
    """
    params = {}
    n_cplx, n_lig = data.shape
    n_rcp = len(rcps)

    idx = 0
    for param_name in PARAM_ORDER:
        idx_end = idx
        if param_name == "Rtot":
            idx_end = idx + n_cplx * n_rcp
            params["Rtot"] = log_x[idx:idx_end].reshape((n_cplx, n_rcp))
        if param_name == "logistic_params":
            n_logist_lig = n_logistic_ligands(logistic_ligands)
            idx_end = idx + n_logist_lig * 4
            params["logistic_params"] = log_x[idx:idx_end].reshape((4, n_logist_lig))
        idx = idx_end

    return scale_up_params(params)


def flatten_params(params: dict):
    """
    Scales down and flattens params.
    """
    params = scale_down_params(params)
    return np.concatenate([params[k].flatten() for k in PARAM_ORDER if k in params])


def initialize_params(
    data: xr.DataArray,
    logistic_ligands: np.ndarray,
    rcps: Collection = DEFAULT_RCPS,
) -> dict:
    """
    Randomly initializes parameters using distributions informed by data.
    """
    params = {}

    log_mean = np.mean(np.nan_to_num(np.log10(data.values + 1)))  # deal with 0s

    Rtot = np.random.uniform(
        10 ** (log_mean - 4),
        10 ** (log_mean + 2),
        (data.sizes["Complex"], len(rcps)),
    )
    params["Rtot"] = Rtot

    n_logist_lig = n_logistic_ligands(logistic_ligands)
    logist_lig_map = logistic_ligand_map(logistic_ligands)
    logistic_params = np.zeros((4, n_logist_lig))
    logistic_params[0] = np.min(data[:, logist_lig_map], axis=0) + 1e-2
    logistic_params[1] = np.max(data[:, logist_lig_map], axis=0)
    logistic_params[2] = np.random.uniform(
        10 ** (log_mean + 3), 10 ** (log_mean + 5), n_logist_lig
    )
    logistic_params[3] = np.random.uniform(1e-1, 1, n_logist_lig)
    params["logistic_params"] = logistic_params

    return params


def scale_down_params(params: dict):
    params = deepcopy(params)
    params["Rtot"] = np.log(params["Rtot"])
    # don't take log of logistic curve inflection slope
    params["logistic_params"][:3] = np.log(params["logistic_params"][:3])
    return params


def scale_up_params(params: dict):
    params = deepcopy(params)
    params["Rtot"] = np.exp(params["Rtot"])
    params["logistic_params"][:3] = np.exp(params["logistic_params"][:3])
    return params


def assemble_jac_sparsity(
    data: xr.DataArray,
    rcps: tuple,
    logistic_ligands: np.ndarray[bool],
):
    """
    Create Jacobian sparsity matrix for optimization.

    Args:
      data: Raw data
      rcps: Antibody types
      intersample_detections: boolean mask indicating which detection reagents
        should be treated as intersample information.
      params: parameters used in optimization (before flattening)

    Returns: Jacobian sparsity matrix.
    """
    n_cplx, n_lig = data.shape
    n_rcp = len(rcps)
    n_logist_lig = n_logistic_ligands(logistic_ligands)
    logist_lig_map = logistic_ligand_map(logistic_ligands)

    # receptor abundance dependencies for a single complex
    lig_rcp_block = np.zeros((n_lig, n_rcp), dtype=int)

    # ligand abundances which follow multivalent binding depend on all receptors
    lig_rcp_block[~logist_lig_map] = np.ones((n_lig - n_logist_lig, n_rcp))

    # ligand abundances which follow a logistic binding curve only depend on the
    # inputs for the binding curve
    lig_rcp_block |= logistic_ligands

    # tile single-complex dependencies across all complexes
    jac_sparsity = np.zeros((n_cplx, n_cplx, n_lig, n_rcp), dtype=int)
    jac_sparsity[np.arange(n_cplx), np.arange(n_cplx)] = lig_rcp_block
    jac_sparsity = np.moveaxis(jac_sparsity, 2, 1)
    jac_sparsity = jac_sparsity.reshape(n_cplx * n_lig, n_cplx * n_rcp)

    # logistic parameter dependencies for a single complex
    logistic_block_flat = np.zeros((n_lig, n_logist_lig), dtype=bool)
    logistic_block_flat[logist_lig_map, np.arange(n_logist_lig)] = 1
    logistic_block = np.zeros((n_lig, n_logist_lig, 4), dtype=int)
    logistic_block[logistic_block_flat] = np.array([1, 1, 1, 1])
    logistic_block = logistic_block.swapaxes(1, 2).reshape((n_lig, n_logist_lig * 4))

    # tile single-complex dependencies across all complexes
    jac_sparsity = np.hstack((jac_sparsity, np.tile(logistic_block, (n_cplx, 1))))

    jac_sparsity_logistic_slope = np.zeros(
        (n_logist_lig, n_cplx * n_rcp + n_logist_lig * 4), dtype=int
    )
    jac_sparsity_logistic_slope[
        np.arange(n_logist_lig),
        np.arange(n_cplx * n_rcp + n_logist_lig * 3, n_cplx * n_rcp + n_logist_lig * 4),
    ] = 1
    jac_sparsity = np.vstack((jac_sparsity, jac_sparsity_logistic_slope))

    return jac_sparsity


def assemble_x_scale(data, rcps, logistic_ligands):
    n_cplx, n_lig = data.shape
    n_rcp = len(rcps)
    params = []
    for param_name in PARAM_ORDER:
        if param_name == "Rtot":
            params.append(np.ones((n_cplx, n_rcp)))
        if param_name == "logistic_params":
            params.append(
                np.full(
                    (n_logistic_ligands(logistic_ligands), 4),
                    np.array([5e-1, 5e-1, 1e-1, 3e-1]),  # empirical
                ).T
            )
    return np.concatenate([param.flatten() for param in params])


def assemble_bounds(data, rcps, logistic_ligands):
    n_cplx, n_lig = data.shape
    n_rcp = len(rcps)
    lower = []
    upper = []
    for param_name in PARAM_ORDER:
        if param_name == "Rtot":
            lower.append(np.full((n_cplx, n_rcp), -np.Inf))
            upper.append(np.full((n_cplx, n_rcp), np.Inf))
        if param_name == "logistic_params":
            # min value of logistic curve maximum is the maximum of the observed
            # max value of logistic curve minimum is the minimum of the observed
            n_logist_lig = n_logistic_ligands(logistic_ligands)
            min_signal = np.min(data[:, logistic_ligand_map(logistic_ligands)], axis=0)
            max_signal = np.max(data[:, logistic_ligand_map(logistic_ligands)], axis=0)
            lower_ll = np.full((4, n_logist_lig), -np.Inf)
            upper_ll = np.full((4, n_logist_lig), np.Inf)
            upper_ll[0] = np.log(min_signal + 2e-2)  # account for possible 0
            lower_ll[1] = np.log(max_signal)
            lower_ll[-1] = 0
            lower.append(lower_ll)
            upper.append(upper_ll)
    bounds = (
        np.concatenate([param.flatten() for param in lower]),
        np.concatenate([param.flatten() for param in upper]),
    )
    assert bounds[0].size == bounds[1].size
    return bounds
