"""
Core function for serology mechanistic tensor factorization
"""
# Base Python
from typing import Collection, Iterable, List, Union

# Extended Python
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import jit
from jax.config import config
from jaxopt import GaussNewton
from scipy.optimize import least_squares
from scipy.linalg import block_diag
from sklearn.decomposition import NMF as non_neg_matrix_factor

# Current Package
from .preprocess import assembleKav, DEFAULT_AB_TYPES

config.update("jax_enable_x64", True)

DEFAULT_FIT_KA_VAL = False
DEFAULT_FC_IDX_VAL = 4

def initializeParams(cube: xr.DataArray, ab_types: Collection=DEFAULT_AB_TYPES) -> List:
    """
    Generate initial guesses for input parameters.

    Args:
        cube: Prepared data in DataArray or np array form.
          Dims: Samples x Receptors x Ags
        lrank: Determines whether we should assume a low-rank structure for the
          abundance matrix. If true, return Sample and Ag matrices, else return
          Abundance matrix
    
    Returns:
        The list of parameters.
    """
    n_ab_types = len(ab_types)
    Ka = assembleKav(cube, ab_types).values
    samp = np.random.uniform(1E-1, 1E3, (cube.shape[0], n_ab_types))
    ag = np.random.uniform(0, 1, (cube.shape[2], n_ab_types))
    abundance = np.einsum("ij,kj->ijk", samp, ag)
    return [abundance, Ka]

def phi(Phi, Rtot, L0, KxStar, Ka, f):
    temp = jnp.einsum("jl,ijk->ilkj", Ka, (1.0 + Phi) ** (f - 1))
    Req = Rtot[:, :, :, np.newaxis] / (1.0 + f * L0 * temp)
    Phi_temp = jnp.einsum("jl,ilkj->ijk", Ka * KxStar, Req)
    assert Phi_temp.shape == Phi.shape
    return Phi_temp

def phi_res(*args):
    return phi(*args) - args[0]

def inferLbound(cube, *args, L0=1e-9, KxStar=1e-12, FcIdx=DEFAULT_FC_IDX_VAL):
    """
        Pass the matrices generated above into polyc, run through each receptor
        and ant x sub pair and store in matrix same size as flatten.
        *args = r_subj, r_ag, kav (when lrank = True) OR abundance, kav (when lrank = False)
        Numbers in args should NOT be log scaled.
    """
    AB_VALENCY, FC_VALENCY = 2, 4
    assert len(args) == 2, "args == [0] abundance, [1] kav"
    Ka = args[1]
    Rtot = args[0].reshape((cube.shape[0], args[0].shape[1], cube.shape[2]))
    Phi = jnp.zeros((cube.shape[0], cube.shape[1], cube.shape[2]))

    if isinstance(KxStar, Iterable):
        KxStarAb, KxStarRcp = KxStar[0], KxStar[1]
    else:
        KxStarAb, KxStarRcp = KxStar, KxStar

    gn = GaussNewton(residual_fun=phi_res, maxiter=100, tol=1e-12)
    Phi_Ab = gn.run(Phi[:, :FcIdx, :], Rtot, L0, KxStarAb, Ka[:FcIdx], AB_VALENCY).params
    Phi_Fc = gn.run(Phi[:, FcIdx:, :], Rtot, L0, KxStarRcp, Ka[FcIdx:], FC_VALENCY).params

    Lbound_Ab = L0 / KxStarAb * ((1.0 + Phi_Ab) ** AB_VALENCY - 1.0)
    Lbound_Rcp = L0 / KxStarRcp * ((1.0 + Phi_Fc) ** FC_VALENCY - 1.0)
    return jnp.concatenate((Lbound_Ab, Lbound_Rcp), axis=1)

def reshapeParams(log_x: np.ndarray, cube, fitKa: bool=DEFAULT_FIT_KA_VAL, ab_types: Collection=DEFAULT_AB_TYPES, as_xarray: bool=False):
    """ Reshapes factor vector, x, into matrices. Inverse operation of flattenParams(). """
    x = jnp.exp(log_x)
    n_subj, n_rec, n_ag = cube.shape
    n_ab = len(ab_types)

    non_ka_params_len = abundance_len = n_subj * n_ag * n_ab 
    abundance = x[:abundance_len].reshape((n_subj * n_ag, n_ab))
    if as_xarray:
        assert isinstance(cube, xr.DataArray), "When returning xarray instances reshapeParams, the `cube` parameter must be a xarray.DataArray instance to provide the axis labels"
        abundance = abundance.reshape((n_subj, n_ab, n_ag))
        abundance = xr.DataArray(abundance, (cube.Sample.values, list(ab_types), cube.Antigen.values), ("Sample", "Antibody", "Antigen"))
    retVal = [abundance]

    if fitKa:   # retrieve Ka from x as well
        ka_len = n_rec * n_ab
        Ka = x[non_ka_params_len:non_ka_params_len+ka_len].reshape(n_rec, n_ab)
        if as_xarray:
            Ka = xr.DataArray(Ka, (cube.Receptor.values, list(ab_types)), ("Receptor", "Antibody"))
        retVal.append(Ka)
    else:
        ka_len = 0
    return retVal

def flattenParams(*args):
    """ Flatten into a parameter vector. Inverse operation of reshapeParams().
    Order: (r_subj, r_ag) / abund, Ka """
    return jnp.log(jnp.concatenate([a.flatten() for a in args]))

def modelLoss(log_x: np.ndarray, cube: Union[xr.DataArray, np.ndarray], Ka: np.ndarray, 
              ab_types: Collection=DEFAULT_AB_TYPES,
              fitKa: bool=DEFAULT_FIT_KA_VAL, L0=1e-9, KxStar=1e-12, FcIdx=DEFAULT_FC_IDX_VAL) -> jnp.ndarray:
    """
    Computes the loss, comparing model output and actual measurements.

    Args:
        x: np array flattened params
        cube: xarray or np array with the shape of the model output
        Ka: fixed Ka, won't be used if fitKa
        nnoneg_idx: getNonnegIdx(cube)
    
    Returns:
        The loss
    """
    params = reshapeParams(log_x, cube, ab_types=ab_types, fitKa=fitKa) 
    if isinstance(cube, xr.DataArray):
        cube = jnp.array(cube)
    Lbound = inferLbound(cube, *(params + ([] if fitKa else [Ka])), L0=L0, KxStar=KxStar, FcIdx=FcIdx)
    scaling_factor = log_x[-1]

    return jnp.nan_to_num(jnp.log(cube) - jnp.log(Lbound) + scaling_factor).flatten()


def optimizeLoss(
    data: xr.DataArray,
    fitKa=DEFAULT_FIT_KA_VAL,
    ab_types: Collection = DEFAULT_AB_TYPES,
    maxiter: int = 10000,
    L0=1e-9,
    KxStar=1e-12,
    FcIdx=DEFAULT_FC_IDX_VAL,
    params: List = None,
):
    """Optimization method to minimize modelLoss() output"""
    n_subj, n_rec, n_ag = data.shape

    if params is None:
        params = initializeParams(data, ab_types=ab_types)
    Ka = params[-1]
    if not fitKa:
        params = params[:-1]
    # flatten params and append scaling factor
    log_x0 = np.append(flattenParams(*params), np.random.rand(1) * 2)
    arrgs = (
        data.values,
        Ka,  # if fitKa this value won't be used
        ab_types,
        fitKa,
        L0,
        KxStar,
        FcIdx,  # L0 and Kx*
    )
    funnc = jit(modelLoss, static_argnums=np.arange(3, 8))

    # Setup a matrix characterizing the block sparsity of the Jacobian
    A = np.ones((n_rec, len(ab_types)), dtype=int)
    jacHand = block_diag(*([A] * n_subj))
    jacHand = np.pad(jacHand, ((0, 0), (0, 1)), constant_values=1)

    opt = least_squares(
        funnc,
        log_x0,
        args=arrgs,
        verbose=1,
        tr_options={"atol": 1e-9, "btol": 1e-9},
        jac_sparsity=jacHand,
    )

    if not fitKa:
        params.append(Ka)
    ctx = {"opt": opt, "init_params": params}
    ret = [opt.x, ctx]
    return ret


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
        model = non_neg_matrix_factor(n_comps, max_iter=1_000)
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
