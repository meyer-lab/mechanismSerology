"""
Core function for serology mechanistic tensor factorization
""" 
# Base Python
from typing import Iterable, List, Union

# Extended Python
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import value_and_grad, jit, grad
from jax.config import config
from scipy.optimize import minimize
from tqdm import tqdm

# Current Package
from .preprocess import assembleKav, DEFAULT_AB_TYPES, prepare_data

config.update("jax_enable_x64", True)

INIT_SCALER = 100
DEFAULT_FIT_KA_VAL = False
DEFAULT_LRANK_VAL = False
DEFAULT_METRIC_VAL = "rtot"

def initializeParams(cube: Union[xr.DataArray, np.ndarray], lrank=DEFAULT_LRANK_VAL, ab_types: Iterable=DEFAULT_AB_TYPES) -> List:
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
    if lrank:       # with low-rank assumption
        samp = np.random.uniform(1E5, 5E5, (cube.shape[0], n_ab_types))
        ag = np.random.uniform(1E5, 5E5, (cube.shape[2], n_ab_types))
        return [samp, ag, Ka]
    else:           # without low-rank assumption
        abundance = np.random.uniform(1E10, 2E10, (cube.shape[0] * cube.shape[2], n_ab_types))
        return [abundance, Ka]

def phi(Phisum, Rtot, L0, KxStar, Ka):
    temp = jnp.einsum("jl,ijk->ilkj", Ka, 1.0 + Phisum)
    Req = Rtot[:, :, :, np.newaxis] / (1.0 + 2.0 * L0 * temp)
    Phisum_n = jnp.einsum("jl,ilkj->ijk", Ka * KxStar, Req)
    assert Phisum_n.shape == Phisum.shape
    return Phisum_n

def inferLbound(cube, *args, lrank=DEFAULT_LRANK_VAL, L0=1e-9, KxStar=1e-12):
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
    Phisum = jnp.zeros((cube.shape[0], cube.shape[1], cube.shape[2]))

    for ii in range(5):
        Phisum_n = phi(Phisum, Rtot, L0, KxStar, Ka)
        Phisum = Phisum_n

    return L0 / KxStar * ((1.0 + Phisum) ** 2 - 1.0)

def reshapeParams(x, cube, lrank=DEFAULT_LRANK_VAL, fitKa=DEFAULT_FIT_KA_VAL, ab_types: Iterable=DEFAULT_AB_TYPES, as_xarray=False):
    """ Reshapes factor vector, x, into matrices. Inverse operation of flattenParams(). """
    x = jnp.exp(x)
    n_subj, n_rec, n_ag = cube.shape
    n_ab = len(ab_types)

    if as_xarray:
        assert isinstance(cube, xr.DataArray), "When returning xarray instances reshapeParams, the `cube` parameter must be a xarray.DataArray instance to provide the axis labels"

    if not lrank:  # abundance as a whole big matrix
        non_ka_params_len = abundance_len = n_subj * n_ag * n_ab 
        abundance = x[:abundance_len].reshape((n_subj * n_ag, n_ab))
        if as_xarray:
            abundance = abundance.reshape((n_subj, n_ab, n_ag))
            abundance = xr.DataArray(
                abundance, 
                dims=("Sample", "Antibody", "Antigen"),
                coords={
                    "Sample": cube.Sample.values,
                    "Antibody": list(ab_types),
                    "Antigen": cube.Antigen.values,
                }
            )
        retVal = [abundance]
    else:   # separate receptor and antigen matrices
        sample_matrix_len = n_subj * n_ab
        ag_matrix_len = n_ag * n_ab
        non_ka_params_len = sample_matrix_len + ag_matrix_len
        r_subj = x[0:sample_matrix_len].reshape(n_subj, n_ab)
        r_ag = x[sample_matrix_len:sample_matrix_len + ag_matrix_len].reshape(n_ag, n_ab)
        if as_xarray:
            r_subj = xr.DataArray(
                r_subj,
                dims=("Sample", "Antibody"),
                coords={
                    "Sample": cube.Sample.values,
                    "Antibody": list(ab_types)
                }
            )
            r_ag = xr.DataArray(
                r_ag, 
                dims=("Antigen", "Antibody"),
                coords={
                    "Antigen": cube.Antigen.values,
                    "Antibody": list(ab_types)
                }
            )
        retVal = [r_subj, r_ag]
    if fitKa:   # retrieve Ka from x as well
        ka_len = n_rec * n_ab
        Ka = x[non_ka_params_len:non_ka_params_len+ka_len].reshape(n_rec, n_ab)
        if as_xarray:
            # get the labels by calling assembleKav. we can't just set the
            # labels to be the receptors of cube because those receptors are
            # reordered in assembleKav
            Ka_schema = assembleKav(cube, ab_types)
            Ka = xr.DataArray(
                Ka,
                dims=("Receptor", "Antibody"),
                coords={
                    "Receptor": Ka_schema.Receptor.values,
                    "Antibody": Ka_schema.Abs.values,
                }
            )
        retVal.append(Ka)
    else:
        ka_len = 0
    # assert correct length, keeping in mind possible scaling factor
    want_len = (non_ka_params_len + ka_len, non_ka_params_len + ka_len + 1) 
    assert x.shape[0] in want_len, f"reshapeParams got x of invalid length. Want: {' OR '.join(map(str, want_len))}, Got: {x.shape[0]}" 
    return retVal

def flattenParams(*args):
    """ Flatten into a parameter vector. Inverse operation of reshapeParams().
    Order: (r_subj, r_ag) / abund, Ka """
    return jnp.log(jnp.concatenate([a.flatten() for a in args]))

def modelLoss(x: np.ndarray, cube: Union[xr.DataArray, np.ndarray], Ka, nonneg_idx,
              ab_types: Iterable=DEFAULT_AB_TYPES, metric=DEFAULT_METRIC_VAL, lrank=DEFAULT_LRANK_VAL,
              fitKa=DEFAULT_FIT_KA_VAL, L0=1e-9, KxStar=1e-12):
    """
    Computes the loss function, comparing model output and actual measurements.

    Args:
        x: np array flattened params
        cube: xarray or np array with the shape of the model output
        Ka: fixed Ka, won't be used if fitKa
        nnoneg_idx: getNonnegIdx(cube, metric)
    
    Returns:
        The loss
    """
    params = reshapeParams(x, cube, ab_types=ab_types, lrank=lrank, fitKa=fitKa)   # one more item there as scale is fine
    if isinstance(cube, xr.DataArray):
        cube = jnp.array(cube)
    if not fitKa:
        params.append(Ka)
    Lbound = inferLbound(cube, *params, lrank=lrank, L0=L0, KxStar=KxStar)
    if metric == 'mean':
        if len(x) % np.sum([p.shape[0] for p in params]) == 1:  # deal with possible scaling factor
            Lbound = Lbound * x[-1]
        diff = jnp.ravel((jnp.log(cube) - jnp.log(Lbound)))[nonneg_idx]
        return jnp.linalg.norm(diff)
    elif metric == "rtot":
        return -calcModalR(jnp.log(cube), jnp.log(Lbound), valid_idx=nonneg_idx)
    else:   # per Receptor or per Ag ("rag")
        r_list = calcModalR(cube, Lbound, axis=(2 if metric == "rag" else 1), valid_idx=nonneg_idx)
        return -(sum(r_list)/len(r_list))

def optimizeLoss(data: xr.DataArray, metric=DEFAULT_METRIC_VAL, lrank=DEFAULT_LRANK_VAL, fitKa=DEFAULT_FIT_KA_VAL,
                 ab_types: Iterable=DEFAULT_AB_TYPES, maxiter=500, verbose=False, retInit=False):
    """ Optimization method to minimize modelLoss() output """
    data = prepare_data(data)
    params = initializeParams(data, lrank=lrank, ab_types=ab_types)
    Ka = params[-1]
    if not fitKa:
        params = params[:-1]
    x0 = flattenParams(*params)
    if metric == "mean":
        x0 = np.append(x0, INIT_SCALER)   # scaling factor
    arrgs = (data.values, 
             np.log(Ka), # if fitKa this value won't be used
             getNonnegIdx(data.values, metric=metric), 
             ab_types, metric, lrank, fitKa,
             1e-9, 1e-12, # L0 and Kx*
             )
    func = jit(value_and_grad(modelLoss), static_argnums=[4, 5, 6, 7])
    opts = {'maxiter': maxiter}

    def hvp(x, v, *argss):
        return grad(lambda x: jnp.vdot(func(x, *argss)[1], v))(x)

    hvpj = jit(hvp, static_argnums=[5, 6, 7, 8])

    saved_params = { "iteration_number" : 0 }

    def callback(xk):
        a, b = func(xk, *arrgs)
        gNorm = np.linalg.norm(b)
        tq.set_postfix(val='{:.2e}'.format(a), g='{:.2e}'.format(gNorm), refresh=False)
        tq.update(1)
        if verbose:
            loss = modelLoss(xk, data.values, np.log(Ka), getNonnegIdx(data.values, metric=metric), 
                            ab_types,
                            metric, lrank, fitKa, 
                            1e-9, 1e-12,  # L0 and Kx*
                            )
            if saved_params["iteration_number"] % 5 == 0:
                print("{:3} | {}".format(
                    saved_params["iteration_number"],
                    loss
                ))
            print("")
        saved_params["iteration_number"] += 1

    with tqdm(total=maxiter, delay=0.1) as tq:
        opt = minimize(func, x0, method="trust-ncg", args=arrgs, hessp=hvpj,
                       callback=callback, jac=True, options=opts)
        print(f"Exit message: {opt.message}")
        print(f"Exit status: {opt.status}")
    if retInit:
        if not fitKa:
            params.append(Ka)
        return opt.x, opt.fun, params
    return opt.x, opt.fun

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
    if metric in ("rtot", "mean"):
        return jnp.where(jnp.ravel(cube) > 0)
    else: # per receptor or per Ag
        i_list = []
        # assume cube has shape Samples x Receptors x Ags
        cube = np.swapaxes(cube, 0, (2 if metric == "rag" else 1))  # else = per Receptor
        for i in range(cube.shape[0]):
            i_list.append(jnp.where(jnp.ravel(cube[i, :]) > 0))
        return i_list
