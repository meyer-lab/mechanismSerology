"""
Core function for serology mechanistic tensor factorization
""" 
# Base Python
from typing import Collection, Iterable, List, Union

# Extended Python
import jax.numpy as jnp
import numpy as np
import xarray as xr
import jax
from jax import value_and_grad, jit, grad
from jax.config import config
from scipy.optimize import minimize
from sklearn.decomposition import NMF as non_neg_matrix_factor
from tqdm import tqdm

# Current Package
from .preprocess import assembleKav, DEFAULT_AB_TYPES, prepare_data

config.update("jax_enable_x64", True)

DEFAULT_FIT_KA_VAL = False
DEFAULT_LRANK_VAL = False
DEFAULT_METRIC_VAL = "mean_rcp"
AB_VALENCY = 2
FC_VALENCY = 4 

def initializeParams(cube: xr.DataArray, lrank: bool=DEFAULT_LRANK_VAL, ab_types: Collection=DEFAULT_AB_TYPES) -> List:
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
    if lrank:       # with low-rank assumption
        return [samp, ag, Ka]
    else:           # without low-rank assumption
        abundance = np.einsum("ij,kj->ijk", samp, ag)
        return [abundance, Ka]

def phi(Phi, Rtot, L0, KxStar, Ka, f):
    temp = jnp.einsum("jl,ijk->ilkj", Ka, (1.0 + Phi) ** (f - 1))
    Req = Rtot[:, :, :, np.newaxis] / (1.0 + f * L0 * temp)
    Phi_temp = jnp.einsum("jl,ilkj->ijk", Ka * KxStar, Req)
    assert Phi_temp.shape == Phi.shape
    return Phi_temp

def inferLbound(cube, *args, lrank=DEFAULT_LRANK_VAL, L0=1e-9, KxStar=1e-12, FcIdx=4):
    """
        Pass the matrices generated above into polyc, run through each receptor
        and ant x sub pair and store in matrix same size as flatten.
        *args = r_subj, r_ag, kav (when lrank = True) OR abundance, kav (when lrank = False)
        Numbers in args should NOT be log scaled.
    """
    if lrank:
        assert len(args) == 3, "args take 1) r_subj, 2) r_ag, 3) kav [when lrank is True]"
        Ka = args[2]
        Rtot = jnp.einsum("ij,kj->ijk", args[0], args[1])
    else:
        assert len(args) == 2, "args take 1) abundance, 2) kav [when lrank is False]"
        Ka = args[1]
        Rtot = args[0].reshape((cube.shape[0], args[0].shape[1], cube.shape[2]))
    Phi = jnp.zeros((cube.shape[0], cube.shape[1], cube.shape[2]))

    if isinstance(KxStar, Iterable):
        KxStarAb, KxStarRcp = KxStar[0], KxStar[1]
    else:
        KxStarAb, KxStarRcp = KxStar, KxStar

    # anti-subclass Abs have valency 2, Fc receptors have valency 4
    for ii in range(5):
        Phi_Ab = phi(Phi[:, :FcIdx, :], Rtot, L0, KxStarAb, Ka[:FcIdx], AB_VALENCY)
        Phi_Rcp = phi(Phi[:, FcIdx:, :], Rtot, L0, KxStarRcp, Ka[FcIdx:], FC_VALENCY)
        Phi = Phi.at[:, :FcIdx, :].set(Phi_Ab)
        Phi = Phi.at[:, FcIdx:, :].set(Phi_Rcp)

    Lbound_Ab = L0 / KxStarAb * ((1.0 + Phi_Ab) ** AB_VALENCY - 1.0)
    Lbound_Rcp = L0 / KxStarRcp * ((1.0 + Phi_Rcp) ** FC_VALENCY - 1.0)
    return jnp.concatenate((Lbound_Ab, Lbound_Rcp), axis=1)

def reshapeParams(log_x: np.ndarray, cube, lrank: bool=DEFAULT_LRANK_VAL, fitKa: bool=DEFAULT_FIT_KA_VAL, ab_types: Collection=DEFAULT_AB_TYPES, as_xarray: bool=False):
    """ Reshapes factor vector, x, into matrices. Inverse operation of flattenParams(). """
    x = jnp.exp(log_x)
    n_subj, n_rec, n_ag = cube.shape
    n_ab = len(ab_types)

    if as_xarray:
        assert isinstance(cube, xr.DataArray), "When returning xarray instances reshapeParams, the `cube` parameter must be a xarray.DataArray instance to provide the axis labels"

    if not lrank:  # abundance as a whole big matrix
        non_ka_params_len = abundance_len = n_subj * n_ag * n_ab 
        abundance = x[:abundance_len].reshape((n_subj * n_ag, n_ab))
        if as_xarray:
            abundance = abundance.reshape((n_subj, n_ab, n_ag))
            abundance = xr.DataArray(abundance, (cube.Sample.values, list(ab_types), cube.Antigen.values), ("Sample", "Antibody", "Antigen"))
        retVal = [abundance]
    else:   # separate receptor and antigen matrices
        sample_matrix_len = n_subj * n_ab
        ag_matrix_len = n_ag * n_ab
        non_ka_params_len = sample_matrix_len + ag_matrix_len
        r_subj = x[0:sample_matrix_len].reshape(n_subj, n_ab)
        r_ag = x[sample_matrix_len:sample_matrix_len + ag_matrix_len].reshape(n_ag, n_ab)
        if as_xarray:
            r_subj = xr.DataArray(r_subj, (cube.Sample.values, list(ab_types)), ("Sample", "Antibody"))
            r_ag = xr.DataArray(r_ag, (cube.Antigen.values, list(ab_types)), ("Antigen", "Antibody"))
        retVal = [r_subj, r_ag]
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

def modelLoss(log_x: np.ndarray, cube: Union[xr.DataArray, np.ndarray], Ka, nonneg_idx, 
              ab_types: Collection=DEFAULT_AB_TYPES, metric: str=DEFAULT_METRIC_VAL, lrank: bool=DEFAULT_LRANK_VAL,
              fitKa: bool=DEFAULT_FIT_KA_VAL, L0=1e-9, KxStar=1e-12) -> jnp.ndarray:
    """
    Computes the loss, comparing model output and actual measurements.

    Args:
        x: np array flattened params
        cube: xarray or np array with the shape of the model output
        Ka: fixed Ka, won't be used if fitKa
        nnoneg_idx: getNonnegIdx(cube, metric)
    
    Returns:
        The loss
    """
    params = reshapeParams(log_x, cube, ab_types=ab_types, lrank=lrank, fitKa=fitKa)   # one more item there as scale is fine
    if isinstance(cube, xr.DataArray):
        cube = jnp.array(cube)
    Lbound = inferLbound(cube, *(params + ([] if fitKa else [Ka])), lrank=lrank, L0=L0, KxStar=KxStar)
    if metric.startswith("mean"):
        if metric.endswith("autoscale"):
            scaling_factor = jnp.nan_to_num(jnp.log(cube) - jnp.log(Lbound))
            if metric.startswith("mean_rcp"):
                scaling_factor = np.mean(np.mean(scaling_factor, axis=2), axis=0) # per receptor scaling factor
                assert scaling_factor.size == cube.shape[1]
                scaling_factor = scaling_factor[:, np.newaxis]
            else:
                scaling_factor = np.mean(scaling_factor)
        elif metric.startswith("mean_rcp") and len(log_x) - np.sum([p.size for p in params]) == cube.shape[1]:
            scaling_factor = log_x[log_x.size - cube.shape[1]:, np.newaxis]
        elif metric == "mean_direct": scaling_factor = 0
        elif metric.startswith("mean") and len(log_x) - np.sum([p.size for p in params]) == 1:
            scaling_factor = log_x[-1]
        else: raise ValueError("invalid metric")
        diff = jnp.ravel((jnp.log(cube) - (jnp.log(Lbound) + scaling_factor)))[nonneg_idx]
        return jnp.linalg.norm(diff)
    elif metric == "rtot":
        return -calcModalR(jnp.log(cube), jnp.log(Lbound), valid_idx=nonneg_idx)
    else:   # per Receptor or per Ag ("rag")
        r_list = calcModalR(cube, Lbound, axis=(2 if metric == "rag" else 1), valid_idx=nonneg_idx)
        return -(sum(r_list)/len(r_list))

def optimizeLoss(data: xr.DataArray, metric=DEFAULT_METRIC_VAL, lrank=DEFAULT_LRANK_VAL, fitKa=DEFAULT_FIT_KA_VAL,
                 ab_types: Collection=DEFAULT_AB_TYPES, maxiter: int=500, retInit: bool=False, L0=1e-9, KxStar=1e-12, data_id=None):
    """ Optimization method to minimize modelLoss() output """
    data = prepare_data(data, data_id=data_id)
    params = initializeParams(data, lrank=lrank, ab_types=ab_types)
    Ka = params[-1]
    if not fitKa:
        params = params[:-1]
    log_x0 = flattenParams(*params)
    if not metric.endswith("autoscale"):
        if metric.startswith("mean_rcp"):
            log_x0 = np.append(log_x0, np.random.rand(data.Receptor.size) * 2) # scaling factor per receptor
        elif metric.startswith("mean"):
            log_x0 = np.append(log_x0, np.random.rand(1) * 2) # scaling factor
    arrgs = (data.values, 
             Ka, # if fitKa this value won't be used
             getNonnegIdx(data.values, metric=metric),
             ab_types, metric, lrank, fitKa,
             L0, KxStar, # L0 and Kx*
             )
    func = jit(value_and_grad(modelLoss), static_argnums=[4, 5, 6, 7, 8, 9])
    opts = {'maxiter': maxiter}

    def hvp(x, v, *argss):
        return grad(lambda x: jnp.vdot(func(x, *argss)[1], v))(x)

    hvpj = jit(hvp, static_argnums=[5, 6, 7, 8, 9, 10])

    def callback(xk):
        a, b = func(xk, *arrgs)
        gNorm = np.linalg.norm(b)
        tq.set_postfix(loss='{:.2e}'.format(a), g='{:.2e}'.format(gNorm), refresh=False)
        tq.update(1)

    with tqdm(total=maxiter, delay=0.1) as tq:
        opt = minimize(func, log_x0, method="trust-ncg", args=arrgs, hessp=hvpj,
                       callback=callback, jac=True, options=opts)
        print(f"Exit message: {opt.message}")
        print(f"Exit status: {opt.status}")
    ret = [opt.x, opt.fun]
    if retInit:
        if not fitKa:
            params.append(Ka)
        ret.append(params)
    return ret

def calcModalR(cube, lbound, axis=-1, valid_idx=None):
    """ Calculate per Receptor or per Ag R """
    if isinstance(cube, xr.DataArray):
        cube = cube.values
    if axis == -1:  # find overall R
        return jnp.corrcoef(jnp.ravel(cube)[valid_idx], jnp.ravel(lbound)[valid_idx])[0, 1]
    # find modal R
    cube = jnp.swapaxes(cube, 0, axis)
    lbound = jnp.swapaxes(lbound, 0, axis)
    r_list = []
    for i in range(cube.shape[0]):
        cube_val = jnp.ravel(cube[i, :])
        lbound_val = jnp.ravel(lbound[i, :])
        cube_idx = jnp.arange(len(cube_val)) if valid_idx is None else valid_idx[i]
        r_list.append(jnp.corrcoef(jnp.log(cube_val[cube_idx]),
                                   jnp.log(lbound_val[cube_idx]))[0, 1])
    return r_list

def getNonnegIdx(cube, metric=DEFAULT_METRIC_VAL):
    """ Generate/save nonnegative indices for cube so index operations seem static for JAX """
    if isinstance(cube, xr.DataArray):
        cube = cube.values
    if metric == "rtot" or metric.startswith("mean"):
        return jnp.where(jnp.ravel(cube) > 0)
    else: # per receptor or per Ag
        i_list = []
        # assume cube has shape Samples x Receptors x Ags
        cube = np.swapaxes(cube, 0, (2 if metric == "rag" else 1))  # else = per Receptor
        for i in range(cube.shape[0]):
            i_list.append(jnp.where(jnp.ravel(cube[i, :]) > 0))
        return i_list

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
        comp_names = np.arange(1, n_comps+1) 
        sample_facs = xr.DataArray(sample_facs, (abundance.Sample.values, abundance.Antibody.values, comp_names), ("Sample", "Antibody", "Component"))
        ag_facs = xr.DataArray(ag_facs, (abundance.Antigen.values, abundance.Antibody.values, comp_names), ("Antigen", "Antibody", "Component"))
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