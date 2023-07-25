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
from .preprocess import assembleKav, HIgGs, DEFAULT_AB_TYPES

DEFAULT_FIT_KA_VAL = False

def initializeParams(data: xr.DataArray, ab_types: Collection=DEFAULT_AB_TYPES) -> List:
    Ka = assembleKav(data, ab_types).values
    log_mean = np.mean(np.nan_to_num(np.log10(data.values + 1))) # deal with 0s
    abundance = np.random.uniform(10 ** (log_mean - 4), 10 ** (log_mean + 2), (data.sizes["Sample"], len(ab_types), data.sizes["Antigen"]))
    return [abundance, Ka]


def phi(Phi, KaRT, fLKa, f):
    # n_samp * n_rcp * n_ag
    Phi_temp = np.sum(KaRT / (1.0 + fLKa * (1.0 + Phi[:, :, np.newaxis, :]) ** (f[:, np.newaxis, np.newaxis] - 1)), axis=2)
    assert Phi_temp.shape == Phi.shape
    return Phi_temp


def custom_root(Rtot: np.ndarray, L0: np.ndarray, KxStar: np.ndarray, Ka: np.ndarray, f: np.ndarray):
    # Precalculate these quantities for speed
    # n_samp * n_rcp * n_ab * n_ag
    KaRT = Ka[np.newaxis, :, :, np.newaxis] * Rtot[:, np.newaxis, :, :] * KxStar[:, np.newaxis, np.newaxis]
    # 1 * n_rcp * n_ab * 1
    fLKa = f[:, np.newaxis, np.newaxis] * L0[:, np.newaxis, np.newaxis] * Ka[np.newaxis, :, :, np.newaxis]

    # Solve for an initial guess by using f = 2
    a = np.sum(fLKa, axis=2)
    b = 1 + a
    # quadratic formula
    f0 = (-b + np.sqrt(b**2 + 4 * a * np.sum(KaRT, axis=2))) / 2 / a

    root = newton(lambda *args : phi(*args) - args[0], f0, maxiter=15000, args=(KaRT, fLKa, f))
    
    assert np.all(root >= 0) 
    return root


def inferLbound(cube: np.ndarray, Rtot: np.ndarray, Ka: np.ndarray, L0: np.ndarray, KxStar: np.ndarray, f: np.ndarray):
    Rtot = Rtot.reshape((cube.shape[0], Rtot.shape[1], cube.shape[2]))
    Phi = custom_root(Rtot, L0, KxStar, Ka, f)
    return L0[:, np.newaxis] / KxStar[:, np.newaxis] * ((1.0 + Phi) ** f[:, np.newaxis] - 1.0) # lbound
    

def flattenParams(*args):
    """ Flatten into a parameter vector. Inverse operation of reshapeParams().
    Order: (r_subj, r_ag) / abund, Ka """
    return np.log(np.concatenate([a.flatten() for a in args]))

def modelLoss(log_x: np.ndarray, data: Union[xr.DataArray, np.ndarray], Ka: np.ndarray, 
              L0: np.ndarray, KxStar: np.ndarray, f: np.ndarray, ab_types: tuple[str], 
              fitKa: Optional[Union[tuple, bool]]=DEFAULT_FIT_KA_VAL) -> np.ndarray:
    """
    Computes the loss, comparing model output and actual measurements.

    Args:
    
    Returns:
        The loss
    """
    params = reshapeParams(log_x, data, ab_types=ab_types, fitKa=fitKa) 
    if fitKa:
        if isinstance(fitKa, tuple):
            Ka = np.copy(Ka)
            Ka[fitKa] = params[-1]
        else:
            Ka = params[-1]
        params = params[:-1]
    if isinstance(data, xr.DataArray):
        data = np.array(data)
    Lbound = inferLbound(data, *(params + [Ka]), L0, KxStar, f)
    res = (np.nan_to_num(np.log(data) - np.log(Lbound), neginf=0, posinf=0)).flatten()
    return res


def optimizeLoss(
    data: xr.DataArray,
    L0: np.ndarray,
    KxStar: np.ndarray,
    f: np.ndarray,
    ab_types: Optional[tuple[str]] = HIgGs,
    fitKa: Optional[Union[tuple, bool]] = DEFAULT_FIT_KA_VAL,
    params: Optional[List] = None,
    ftol: float = 1e-7,
    gtol: float = 1e-7,
    xtol: float = 1e-7,
) -> Tuple[np.ndarray, Dict]:
    """Optimization method to minimize modelLoss() output"""
    n_samp, n_rcp, n_ag = data.shape
    n_ab = len(ab_types)
    assert L0.shape == (n_rcp,), "L0 wrong shape"
    assert KxStar.shape == (n_rcp,), "KxStar wrong shape"
    assert f.shape == (n_rcp,), "f wrong shape"

    if params is None:
        params = initializeParams(data, ab_types=ab_types)
    Ka = params[-1]
    if fitKa: 
        if isinstance(fitKa, tuple):
            # fit specific elements of Ka
            params[-1] = params[-1][fitKa]
        # if fitKa is True, fit the entire Ka matrix
    else:
        params = params[:-1]

    log_x0 = flattenParams(*params)

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
        modelLoss,
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

def reshapeParams(log_x: np.ndarray, data, fitKa: Optional[Union[tuple, bool]]=DEFAULT_FIT_KA_VAL, ab_types: Collection=DEFAULT_AB_TYPES, as_xarray: bool=False):
    """ Reshapes factor vector, x, into matrices. Inverse operation of flattenParams(). """
    x = np.exp(log_x)
    n_subj, n_rec, n_ag = data.shape
    n_ab = len(ab_types)

    non_ka_params_len = abundance_len = n_subj * n_ag * n_ab 
    abundance = x[:abundance_len].reshape((n_subj * n_ag, n_ab))
    if as_xarray:
        assert isinstance(data, xr.DataArray), "When returning xarray instances reshapeParams, the `cube` parameter must be a xarray.DataArray instance to provide the axis labels"
        abundance = abundance.reshape((n_subj, n_ab, n_ag))
        abundance = xr.DataArray(abundance, (data.Sample.values, list(ab_types), data.Antigen.values), ("Sample", "Antibody", "Antigen"))
    retVal = [abundance]

    if fitKa:   # retrieve Ka from x as well
        if isinstance(fitKa, tuple):
            # return fitted affinities as 1d array
            ka_len = len(fitKa[0])
            Ka = x[non_ka_params_len:non_ka_params_len+ka_len]
        else:
            # return entire Ka matrix
            ka_len = n_rec * n_ab
            Ka = x[non_ka_params_len:non_ka_params_len+ka_len].reshape(n_rec, n_ab)
            if as_xarray:
                Ka = xr.DataArray(Ka, (data.Receptor.values, list(ab_types)), ("Receptor", "Antibody"))
        retVal.append(Ka)
    return retVal
