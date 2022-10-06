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
from .preprocess import prepare_data, assemble_Kav
from jax.config import config


config.update("jax_enable_x64", True)

def initializeParams(cube, lrank=True, fitKa=True, n_ab=4):
    """
        Generate initial guesses for input parameters.
        cube = Samples x Receptors x Ags
        lrank: whether assume a low-rank structure.
            Return separate Subj and Ag matrices if true, otherwise return just one abundance matrix
        fitKa: if Ka is not fix, return a random Ka matrix too
    """
    if fitKa:     # when Ka matrix is not fixed
        Ka = np.random.uniform(1E5, 5E5, (cube.shape[1], n_ab))
    if lrank:       # with low-rank assumption
        samp = np.random.uniform(1E5, 5E5, (cube.shape[0], n_ab))
        ag = np.random.uniform(1E5, 5E5, (cube.shape[2], n_ab))
        return [samp, ag, Ka] if fitKa else [samp, ag]
    else:           # without low-rank assumption
        abundance = np.random.uniform(1E10, 2E10, (cube.shape[0] * cube.shape[2], n_ab))
        return [abundance, Ka] if fitKa else [abundance]

def phi(Phisum, Rtot, L0, KxStar, Ka):
    temp = jnp.einsum("jl,ijk->ilkj", Ka, 1.0 + Phisum)
    Req = Rtot[:, :, :, np.newaxis] / (1.0 + 2.0 * L0 * temp)
    Phisum_n = jnp.einsum("jl,ilkj->ijk", Ka * KxStar, Req)
    assert Phisum_n.shape == Phisum.shape
    return Phisum_n

def inferLbound(cube, *args, lrank=True, L0=1e-9, KxStar=1e-12):
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

def reshapeParams(x, cube, lrank=True, fitKa=True):
    """ Reshapes factor vector, x, into matrices. Inverse operation of flattenParams(). """
    x = jnp.exp(x)
    n_subj, n_rec, n_ag = cube.shape
    edge_size = np.sum(cube.shape) if fitKa else ((n_subj + n_ag) if lrank else (n_subj * n_ag))
    n_ab = int(len(x) / edge_size)
    if not lrank:  # abundance as a whole big matrix
        abundance = x.reshape((n_subj * n_ag, n_ab))
        retVal = [abundance]
    else:   # separate receptor and antigen matrices
        r_subj = x[0:(n_subj * n_ab)].reshape(n_subj, n_ab)
        r_ag = x[(n_subj * n_ab):((n_subj + n_ag) * n_ab)].reshape(n_ag, n_ab)
        retVal = [r_subj, r_ag]
    if fitKa:   # retrieve Ka from x as well
        Ka = x[(n_subj + n_ag) * n_ab:(n_subj + n_ag + n_rec) * n_ab].reshape(n_rec, n_ab)
        retVal.append(Ka)
    return retVal

def flattenParams(*args):
    """ Flatten into a parameter vector. Inverse operation of reshapeParams().
    Order: (r_subj, r_ag) / abund, Ka """
    return jnp.log(jnp.concatenate([a.flatten() for a in args]))

def modelLoss(x, cube, metric="mean", lrank=True, fitKa=True, L0=1e-9, KxStar=1e-12, *args):
    """
        Loss function, comparing model output and actual values.
        args:
            [0]: KaFixed, won't be used if fitKa
            [1]: getNonnegIdx(cube, metric)
    """
    params = reshapeParams(x, cube, lrank=lrank, fitKa=fitKa)   # one more item there as scale is fine
    if isinstance(cube, xr.DataArray):
        cube = jnp.array(cube)
    if not fitKa:
        params.append(args[0])
    Lbound = inferLbound(cube, *params, lrank=lrank, L0=L0, KxStar=KxStar)
    if metric == 'mean':
        mask = (cube > 0)
        if len(x) % np.sum([p.shape[0] for p in params]) == 1:  # deal with possible scaling factor
            Lbound = Lbound * x[-1]
        diff = (jnp.log(cube) - jnp.log(Lbound)) * mask
        return jnp.linalg.norm(diff)
    elif metric == "rtot":
        return -jnp.corrcoef(jnp.ravel(cube)[args[1]],
                             jnp.ravel(Lbound)[args[1]])[0,1]
    else:   # per Receptor or per Ag ("rag")
        axis = (2 if metric == "rag" else 1)
        cube = jnp.swapaxes(cube, 0, axis)
        lbound = jnp.swapaxes(Lbound, 0, axis)
        r_list = []
        for i in range(cube.shape[0]):
            cube_val = jnp.ravel(cube[i, :])
            lbound_val = jnp.ravel(lbound[i, :])
            cube_idx = args[1][i]
            r_list.append(jnp.corrcoef(jnp.log(cube_val[cube_idx]),
                                       jnp.log(lbound_val[cube_idx]))[0, 1])
        return -(sum(r_list)/len(r_list))


def optimizeLoss(data: xr.DataArray, metric="mean", lrank=True, fitKa=False,
                 n_ab=1, maxiter=500, verbose=False, fucose=False):
    """ Optimization method to minimize modelLoss() output """
    data = prepare_data(data)
    KaFixed = assemble_Kav(data, fucose=fucose)   # if fitKa this value won't be used
    assert np.all(KaFixed > 0)
    if not fitKa:     # if not fitKa the input n_ab is useless
        n_ab = KaFixed.shape[1]
    params = initializeParams(data, lrank=lrank, fitKa=fitKa, n_ab=n_ab)
    x0 = flattenParams(*params)
    if metric == "mean":
        x0 = np.append(x0, 1E2)   # scaling factor

    arrgs = (data.values, metric, lrank, fitKa,
             1e-9, 1e-12,    # L0 and KxStar
             np.log(KaFixed).values,   # if fitKa this value won't be used
             getNonnegIdx(data.values, metric=metric))
    func = jit(value_and_grad(modelLoss), static_argnums=[2, 3, 4])
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
        if verbose:
            if saved_params["iteration_number"] % 5 == 0:
                print("{:3} | {}".format(
                    saved_params["iteration_number"],
                    modelLoss(xk, data.values, metric, lrank, fitKa, 1e-9, 1e-12,
                              jnp.log(KaFixed.values),
                              getNonnegIdx(data.values, metric=metric))
                ))
            print("")
        saved_params["iteration_number"] += 1

    with tqdm(total=maxiter, delay=0.1) as tq:
        opt = minimize(func, x0, method="trust-ncg", args=arrgs, hessp=hvpj,
                       callback=callback, jac=True, options=opts)
        print(f"Exit message: {opt.message}")
        print(f"Exit status: {opt.status}")
    return opt.x, opt.fun


def getNonnegIdx(cube, metric="rtot"):
    """ Generate/save nonnegative indices for cube so index operations seem static for JAX """
    if isinstance(cube, xr.DataArray):
        cube = cube.values
    if metric == "rtot":
        return jnp.where(jnp.ravel(cube) > 0)
    else:
        i_list = []
        # assume cube has shape Samples x Receptors x Ags
        cube = np.swapaxes(cube, 0, (2 if metric == "rag" else 1))  # else = per Receptor
        for i in range(cube.shape[0]):
            i_list.append(jnp.where(jnp.ravel(cube[i, :]) > 0))
        return i_list
