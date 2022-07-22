import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from scipy.optimize import minimize
from jax import value_and_grad, jit, grad
from .predictAbundKa import infer_Lbound


def initial_subj_abund(cube, n_ab=1):
    """
    Generate subjects and Ka matrices by initializing all values to 10**6
    cube.shape == n_subj * n_rec * antigen
    """
    # TODO: Add masking
    subj_matrix = np.full((cube.shape[0], n_ab), 10**6)
    ag_matrix = np.full((cube.shape[2], n_ab), 10**6)
    return subj_matrix, ag_matrix

def flatten_params(r_subj, r_ag):
    """ Flatten into a parameter vector"""
    return np.log(np.concatenate((r_subj.flatten(), r_ag.flatten())))

def reshapeParams(x, cube):
    """ Unflatten parameter vector back into subject and antigen matrices """
    x = jnp.exp(x)
    n_subj, n_rec,  n_Ag = cube.shape
    n_ab = int(len(x) / (n_subj + n_Ag))
    R_subj = x[0:(n_subj * n_ab)].reshape(n_subj, n_ab)
    R_Ag = x[(n_subj * n_ab):((n_subj + n_Ag) * n_ab)].reshape(n_Ag, n_ab)
    return R_subj, R_Ag

def model_lossfunc(x, cube, Ka, L0=1e-9, KxStar=1e-12):
    """
    Loss function, comparing model output and flattened tensor
    """
    R_subj, R_Ag = reshapeParams(x, cube)
    Lbound = infer_Lbound(R_subj, R_Ag, Ka, L0=L0, KxStar=KxStar)
    diff = jnp.log(cube) - jnp.log(Lbound)
    diff -= jnp.nanmean(diff)
    return jnp.linalg.norm(jnp.nan_to_num(diff))

def optimize_lossfunc(cube, Kav, n_ab=1, maxiter=100):
    """
    Optimization method to minimize model_lossfunc output
    """
    R_subj_guess, R_Ag_guess = initial_subj_abund(cube, n_ab=n_ab)
    Kav[np.where(Kav==0.0)] = 10
    Kav_log = np.log(Kav)
    x0 = flatten_params(R_subj_guess, R_Ag_guess)

    func = jit(value_and_grad(model_lossfunc))
    opts = {'maxiter': maxiter}
    arrgs = (cube, Kav_log, 1e-9, 1e-12)

    def hvp(x, v, *argss):
        return grad(lambda x: jnp.vdot(func(x, *argss)[1], v))(x)

    hvpj = jit(hvp)

    with tqdm(total=maxiter, delay=0.1) as tq:
        def callback(xk):
            a, b = func(xk, *arrgs)
            gNorm = np.linalg.norm(b)
            tq.set_postfix(val='{:.2e}'.format(a), g='{:.2e}'.format(gNorm), refresh=False)
            tq.update(1)

        print("")

        opt = minimize(func, x0, method="trust-ncg", args=arrgs, hessp=hvpj, callback=callback, jac=True, options=opts)
    return opt.x[:, np.newaxis]

