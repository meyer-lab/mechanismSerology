"""
Currently runs polyfc to compare with SpaceX data with 6 receptors, 117 subjects, and 14 antigens.
Polyfc is ran with initial guesses for abundance (117x1) and (14x1)  and Ka for each receptor (6x1).
Polyfc output is compared to SpaceX data and cost function is minimzied through scipy.optimize.minimize.
Total fitting parameters = 147
""" 
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import xarray as xr
from scipy.optimize import minimize
from jax import value_and_grad, jit, grad
from .data_preparation import prepare_data, assemble_Kav
from .fixkav_opt_helpers import calculate_r_list_from_index, get_indices
from jax.config import config


config.update("jax_enable_x64", True)

def initializeParams(cube, lrank=True, retKa=True, n_ab=1):
    """
        Generate initial guesses for input parameters.
        cube = Subjs x Receptors x Ags
        lrank: whether assume a low-rank structure.
            Return separate Subj and Ag matrices if do, otherwise return just one abundance matrix
        retKa: if Ka is not fix, return a random Ka matrix too
    """
    if (retKa):     # when Ka matrix is not fixed
        Ka = np.random.uniform(1E5, 5E5, (cube.shape[1], n_ab))
    if lrank:       # with low-rank assumption
        subj = np.random.uniform(1E5, 5E5, (cube.shape[0], n_ab))
        ag = np.random.uniform(1E5, 5E5, (cube.shape[2], n_ab))
        return [subj, ag, Ka] if retKa else [subj, ag]
    else:           # without low-rank assumption
        abundance = np.random.uniform(1E10, 2E10, (cube.shape[0] * cube.shape[2], n_ab))
        return [abundance, Ka] if retKa else [abundance]

def phi(Phisum, Rtot, L0, KxStar, Ka):
    temp = jnp.einsum("jl,ijk->ilkj", Ka, 1.0 + Phisum)
    Req = Rtot[:, :, :, np.newaxis] / (1.0 + 2.0 * L0 * temp)
    Phisum_n = jnp.einsum("jl,ilkj->ijk", Ka * KxStar, Req)
    assert Phisum_n.shape == Phisum.shape
    return Phisum_n

def infer_Lbound(cube, *args, lrank=True, L0=1e-9, KxStar=1e-12):
    """
    Pass the matrices generated above into polyc, run through each receptor
    and ant x sub pair and store in matrix same size as flatten.
    *args = r_subj, r_ag, kav (when lrank = True) OR abundance, kav (when lrank = False)
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

def reshapeParams(x, cube, lrank=True, retKa=True):
    """ Reshapes factor vector, x, into matrices. Inverse operation of flattenParams(). """
    x = jnp.exp(x)
    n_subj, n_rec, n_ag = cube.shape
    n_ab = int(len(x) / (jnp.sum(cube.shape) if retKa else ((n_subj + n_ag) if lrank else (n_subj * n_ag))))
    if (retKa):
        Ka = x[(n_subj + n_ag) * n_ab:(n_subj + n_ag + n_rec) * n_ab].reshape(n_rec, n_ab)
    if not lrank:
        abundance = x.reshape((n_subj * n_ag, n_ab))
        return [abundance, Ka] if retKa else [abundance]
    r_subj = x[0:(n_subj * n_ab)].reshape(n_subj, n_ab)
    r_ag = x[(n_subj * n_ab):((n_subj + n_ag) * n_ab)].reshape(n_ag, n_ab)
    return [r_subj, r_ag, Ka] if retKa else [r_subj, r_ag]

def flattenParams(*args):
    """ Flatten into a parameter vector. Inverse operation of reshapeParams(). """
    return jnp.log(jnp.concatenate([a.flatten() for a in args]))

def model_lossfunc(x, cube, metric, lrank=True, retKa=True, L0=1e-9, KxStar=1e-12, *args):
    """ Loss function, comparing model output and actual values. """
    arr = x[:-1]
    scale = x[-1]

    params = reshapeParams(arr, cube, lrank=lrank, retKa=retKa)
    if not retKa: params.append(args[0])
    Lbound = infer_Lbound(cube, *params, lrank=lrank, L0=L0, KxStar=KxStar)
    if (metric == 'mean'):
        mask = (cube > 0)
        Lbound = Lbound * scale
        diff = (jnp.log10(cube) - jnp.log10(Lbound)) * mask
        return jnp.linalg.norm(diff)
    else:
        cube_flat = jnp.ravel(cube)[args[2]]
        lbound_flat = jnp.ravel(Lbound)[args[2]]
        non_nan = (~jnp.isnan(cube_flat))
        if (metric == 'rtot'): 
            return -jnp.corrcoef(cube_flat * non_nan, lbound_flat * non_nan) [0,1]
        elif (metric == 'r'):
            r_list = calculate_r_list_from_index(cube_flat * non_nan, lbound_flat * non_nan, args[1])
            return -(sum(r_list)/len(r_list))


def optimize_lossfunc(data: xr.DataArray, metric, lrank=True, retKav=True, perReceptor=True, n_ab=1, maxiter=500):
    """ Optimization method to minimize model_lossfunc output """
    data = prepare_data(data)
    kav = None if retKav else assemble_Kav(data)
    if kav.all() : kav_log = np.log(kav)
    params = initializeParams(data, lrank=lrank, retKa=retKav, n_ab=n_ab)
    x0 = flattenParams(*params)
    x0 = np.append(x0, 1E2) # scaling factor
    data_flat = jnp.ravel(data.values)

    arrgs = (data.values, metric, lrank, retKav, 1e-9, 1e-12, kav_log.values, get_indices(data, perReceptor), jnp.nonzero(data_flat))
    func = jit(value_and_grad(model_lossfunc), static_argnums=[2, 3, 4])
    opts = {'maxiter': maxiter}

    def hvp(x, v, *argss):
        return grad(lambda x: jnp.vdot(func(x, *argss)[1], v))(x)

    hvpj = jit(hvp, static_argnums=[3, 4, 5])

    saved_params = { "iteration_number" : 0 }
    def callback(xk):
        a, b = func(xk, *arrgs)
        gNorm = np.linalg.norm(b)
        tq.set_postfix(val='{:.2e}'.format(a), g='{:.2e}'.format(gNorm), refresh=False)
        tq.update(1)
        if saved_params["iteration_number"] % 5 == 0:
            print("{:3} | {}".format(
            saved_params["iteration_number"], model_lossfunc(xk, data.values, metric, lrank, retKav, 1e-9, 1e-12, kav_log.values, get_indices(data, perReceptor), jnp.nonzero(data_flat))))
        saved_params["iteration_number"] += 1
        print("")

    with tqdm(total=maxiter, delay=0.1) as tq:
        opt = minimize(func, x0, method="trust-ncg", args=arrgs, hessp=hvpj, callback=callback, jac=True, options=opts)
        print(f"Exit message: {opt.message}")
        print(f"Exit status: {opt.status}")

    return opt.x[:, np.newaxis], opt.fun

