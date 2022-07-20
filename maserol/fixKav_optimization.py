import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from scipy.optimize import minimize
from jax import value_and_grad, jit, jacfwd, jacrev
from maserol.predictAbundKa import infer_Lbound
import maserol.model as model

def initial_subj_abund(cube, n_ab=1):
    """
    Generate subjects and Ka matrices by initializing all values to 10**6
    cube.shape == n_subj * n_rec * antigen
    """
    subj_matrix = np.full((cube.shape[0], n_ab), 1000)
    ag_matrix = np.full((cube.shape[2], n_ab), 1000)
    return subj_matrix, ag_matrix

def flatten_params(r_subj, r_ag):
    """ Flatten into a parameter vector"""
    return np.log(np.concatenate((r_subj.flatten(), r_ag.flatten())))

def reshapeParams(x, cube):
    """ Unflatten parameter vector back into subject and antigen matrices """
    x = jnp.exp(x)
    n_subj, n_rec, n_Ag = cube.shape
    n_ab = int(len(x) / (n_subj + n_Ag))
    R_subj = x[0:(n_subj * n_ab)].reshape(n_subj, n_ab)
    R_Ag = x[(n_subj * n_ab):((n_subj + n_Ag) * n_ab)].reshape(n_Ag, n_ab)
    return R_subj, R_Ag

def model_lossfunc(x, cube, Ka, cube_nonzero, use_r= False, L0=1e-9, KxStar=1e-12):
    """
       Inputs: 
        x - flattened parameter fector
        cube - raw data
        Ka - matrix of known affinity values
        cube_nonzero - list of indices where flattened cube has values of zero
        use_r2 - a boolean that is True if r is being used to evaluate the loss
                 false of difference is being used

    Loss function, comparing model output and flattened tensor
    """
    R_subj, R_Ag = reshapeParams(x, cube)
    Lbound = infer_Lbound(R_subj, R_Ag, Ka, L0=L0, KxStar=KxStar)
    if (use_r == True):
        # remove values from lbound where the cube had missing data (zeros)
        Lbound_flat = jnp.ravel(Lbound)[cube_nonzero]
        cube_flat = jnp.ravel(cube)[cube_nonzero]
        
        # calcualte correlation coefficient (r)
        corr_matrix = jnp.corrcoef(cube_flat, Lbound_flat)
        corr = corr_matrix[0,1]
        return -corr
    else:
        mask = (cube > 0)
        diff = ((jnp.log(cube) - jnp.log(Lbound)) * mask) 
        diff -= jnp.mean(diff)
        return jnp.linalg.norm(diff)

def optimize_lossfunc(cube, Kav, use_r, n_ab=1, maxiter=100):
    """
    Optimization method to minimize model_lossfunc output
    """
    cube_nonzero = np.nonzero(jnp.ravel(cube))
    R_subj_guess, R_Ag_guess = initial_subj_abund(cube, n_ab=n_ab)
    x0 = flatten_params(R_subj_guess, R_Ag_guess)

    func = jit(value_and_grad(model_lossfunc), static_argnums=[4])
    hess = jit(jacfwd(jacrev(model_lossfunc)), static_argnums=[4])
    opts = {'maxiter': maxiter}
    arrgs = (cube, Kav, cube_nonzero, use_r, 1e-9, 1e-12)

    with tqdm(total=maxiter, delay=0.1) as tq:
        saved_params = { "iteration_number" : 0 }
        def callback(xk):
            a, b = func(xk, *arrgs)
            gNorm = np.linalg.norm(b)
            tq.set_postfix(val='{:.2e}'.format(a), g='{:.2e}'.format(gNorm), refresh=False)
            tq.update(1)
            if saved_params["iteration_number"] % 5 == 0:
                print("{:3} | {}".format(
                saved_params["iteration_number"], model_lossfunc(xk, cube, Kav, cube_nonzero, use_r)))
            saved_params["iteration_number"] += 1

        print("")

        opt = minimize(func, x0, method="trust-ncg", args=arrgs, hess=hess, callback=callback, jac=True, options=opts)
    return opt.x[:, np.newaxis]
