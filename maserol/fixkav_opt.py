from jax import value_and_grad, jit, jacfwd, jacrev
from model import prepare_data, assemble_Kavf
from predictAbundKa import infer_Lbound
from scipy.optimize import minimize
from tqdm import tqdm
import jax.numpy as jnp
import numpy as np
import xarray as xr

def initial_subj_abund(cube, n_ab=1):
    """Generate subjects and Ka matrices by initializing all values to a random value between 8E8 and 15E8. """
    subj_matrix = np.random.uniform(8E8, 15E8, (cube.shape[0], n_ab))
    ag_matrix = np.random.uniform(8E8, 15E8, (cube.shape[2], n_ab))
    return subj_matrix, ag_matrix

def flatten_params(r_subj, r_ag):
    """ Flatten r_subj and r_ag into a parameter vector."""
    return np.log(np.concatenate((r_subj.flatten(), r_ag.flatten())))

def reshape_params(x, cube):
    """ Unflatten parameter vector back into subject and antigen matrices. """
    x = jnp.exp(x)
    n_subj, n_rec, n_ag = cube.shape
    n_ab = int(len(x) / (n_subj + n_ag))
    r_subj = x[0:(n_subj * n_ab)].reshape(n_subj, n_ab)
    r_ag = x[(n_subj * n_ab):((n_subj + n_ag) * n_ab)].reshape(n_ag, n_ab)
    return r_subj, r_ag

def model_lossfunc(x, cube, kav, cube_nonzero, use_r=False, L0=1e-9, KxStar=1e-12):
    """
    Inputs: 
        x - flattened parameter fector
        cube - raw data
        kav - matrix of known affinity values
        cube_nonzero - list of indices where flattened cube has a value of zero
        use_r2 - a boolean that is 'True' if r (pearson correlation) is being used
                 to evaluate the loss, and 'False' of mean sqaured is being used
                
    Loss function, comparing model output and flattened tensor.
    """
    r_subj, r_ag = reshape_params(x, cube)
    Lbound = infer_Lbound(r_subj, r_ag, kav, L0=L0, KxStar=KxStar)
    if (use_r == True):
        # remove values from lbound where the cube has missing data (zeros)
        Lbound_flat = jnp.ravel(Lbound)[cube_nonzero]
        cube_flat = jnp.ravel(cube)[cube_nonzero]
        
        corr_matrix = jnp.corrcoef(cube_flat, Lbound_flat)
        return -corr_matrix[0,1]
    else:
        mask = (cube > 0)
        diff = ((jnp.log(cube) - jnp.log(Lbound)) * mask) 
        diff -= jnp.mean(diff)
        return jnp.linalg.norm(diff)

def optimize_lossfunc(data: xr.DataArray, kav, use_r=False, n_ab=1, maxiter=500):
    """
    Optimization method to minimize model_lossfunc output
    """
    nonzero_indices = np.nonzero(jnp.ravel(data.values))
    r_subj_guess, r_ag_guess = initial_subj_abund(data.values, n_ab=n_ab)
    x0 = flatten_params(r_subj_guess, r_ag_guess)

    func = jit(value_and_grad(model_lossfunc), static_argnums=[4])
    hess = jit(jacfwd(jacrev(model_lossfunc)), static_argnums=[4])
    opts = {'maxiter': maxiter}
    arrgs = (data.values, kav, nonzero_indices, use_r, 1e-9, 1e-12)

    with tqdm(total=maxiter, delay=0.1) as tq:
        saved_params = { "iteration_number" : 0 }
        def callback(xk):
            a, b = func(xk, *arrgs)
            gNorm = np.linalg.norm(b)
            tq.set_postfix(val='{:.2e}'.format(a), g='{:.2e}'.format(gNorm), refresh=False)
            tq.update(1)
            if saved_params["iteration_number"] % 5 == 0:
                print("{:3} | {}".format(
                saved_params["iteration_number"], model_lossfunc(xk, data.values, kav, use_r)))
            saved_params["iteration_number"] += 1

        print("")

        opt = minimize(func, x0, method="trust-ncg", args=arrgs, hess=hess, callback=callback, jac=True, options=opts)
        print(f"Exit message: {opt.message}")
        print(f"Exit status: {opt.status}")

    return opt.x[:, np.newaxis]

def run_optimization(data: xr.DataArray, use_r=False, abs=1):
    data = prepare_data(data)
    kav = assemble_Kavf(data)
    kav_log = np.log(kav)
    final_matrix = optimize_lossfunc(data, kav_log.values, use_r=use_r, n_ab=abs)
    r_subj_pred, r_ag_pred = reshape_params(final_matrix, data)
    lbound = infer_Lbound(r_subj_pred, r_ag_pred, kav_log.values)
    return r_subj_pred, r_ag_pred, lbound, kav # returns not log version of everything(!)

if __name__ == "__main__":
    pass