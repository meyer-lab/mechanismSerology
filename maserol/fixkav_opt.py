from mechanismSerology.maserol.model import prepare_data, assemble_Kavf
from jax import value_and_grad, jit, grad, jacfwd, jacrev
from mechanismSerology.maserol.predictAbundKa import infer_Lbound
from mechanismSerology.maserol.fixkav_opt_helpers import calculate_r_list_from_index, get_indices
from scipy.optimize import minimize
from tqdm import tqdm
import jax.numpy as jnp
import numpy as np
import xarray as xr

def initialize_params(cube, lrank=False, n_ab=1,):
    """Generate initial guesses for input parameters."""
    if lrank:
        subj_matrix = np.random.uniform(1E5, 5E5, (cube.shape[0], n_ab))
        ag_matrix = np.random.uniform(1E5, 5E5, (cube.shape[2], n_ab))
        return subj_matrix, ag_matrix
    else:
        abundance_matrix = np.random.uniform(1E10, 2E10, (cube.shape[0] * cube.shape[2], n_ab))
        return abundance_matrix

def flatten_params(lrank=False, **kwargs):
    """ Flatten initialized matrices into a parameter vector."""
    if lrank:
        return np.log(np.concatenate((kwargs['r_subj'].flatten(), kwargs['r_ag'].flatten())))
    else:
        return np.log(kwargs['r_abund']).flatten()

def reshape_params(x, cube, lrank=False):
    """ Unflatten parameter vector back into matrix """
    x = jnp.exp(x)
    n_subj, n_rec, n_ag = cube.shape
    if lrank:
        n_ab = int(len(x) / (n_subj + n_ag))
        r_subj = x[0:(n_subj * n_ab)].reshape(n_subj, n_ab)
        r_ag = x[(n_subj * n_ab):((n_subj + n_ag) * n_ab)].reshape(n_ag, n_ab)
        return r_subj, r_ag
    else:
        n_ab = int(len(x) / (n_subj * n_ag))
        abundance_matrix = x.reshape((n_subj * n_ag, n_ab))
        return abundance_matrix


def model_lossfunc(x, cube, kav, metric, lrank=False, L0=1e-9, KxStar=1e-12, *args):
    """
    Inputs: 
        x - flattened parameter fector
        cube - raw data
        input - either known affinity or abundance depending on what is being fit
        metric - r, rtot, mean
        lrank - when True, performs low rank optimization, when False performs
                high rank optimization
        args - can have up to two arguments representing the indices of the nonzero values
               in cube, and a matrix of the indices of each receptor present in the cube
                
    Loss function comparing model output and flattened tensor.
    """
    arr = x[:-1]
    scale = x[-1]
    
    if (lrank):
        r_subj, r_ag = reshape_params(arr, cube, lrank=lrank)
        Lbound = infer_Lbound(cube, kav, lrank=lrank, L0=L0, KxStar=KxStar, r_subj=r_subj, r_ag=r_ag)
    else:
        r_abund = reshape_params(arr, cube, lrank=lrank)
        Lbound = infer_Lbound(cube, kav, lrank=lrank, L0=L0, KxStar=KxStar, r_abund=r_abund)

    if (metric == 'mean'):
        mask = (cube > 0)
        Lbound = Lbound * scale  
        diff = ((jnp.log(cube) - jnp.log(Lbound)) * mask)  
        return jnp.linalg.norm(diff)
    else:
        Lbound_flat = jnp.ravel(Lbound)[args[0]]
        cube_flat = jnp.ravel(cube)[args[0]]

        if (metric == 'rtot'):
            corr_matrix = jnp.corrcoef(cube_flat, Lbound_flat)
            return -corr_matrix[0,1]
        
        elif (metric == 'r'):
            r_list = calculate_r_list_from_index(cube_flat, Lbound_flat, args[1], False)
            avg = -(sum(r_list)/len(r_list))
            return avg

def optimize_lossfunc(data: xr.DataArray, kav, metric, lrank=False, per_receptor= False, n_ab=1, maxiter=1000):
    """
    Optimization method to minimize model_lossfunc output
    """
    nonzero_indices = []
    r_index_matrix = []
    data = prepare_data(data)

    if lrank:
        r_subj_guess, r_ag_guess = initialize_params(data.values, lrank=True, n_ab=n_ab)
        array = flatten_params(lrank, r_subj=r_subj_guess, r_ag=r_ag_guess)
    else:
        r_abund_guess = initialize_params(data.values, lrank=False, n_ab=n_ab)
        array = flatten_params(lrank, r_abund=r_abund_guess)

    x0 = np.append(array, 1E2)
    
    if (metric != 'mean'):
        nonzero_indices = np.nonzero(jnp.ravel(data.values))
        r_index_matrix = get_indices(data, per_receptor)
    arrgs = (data.values, kav, metric, lrank, 1e-9, 1e-12, nonzero_indices, r_index_matrix)

    func = jit(value_and_grad(model_lossfunc), static_argnums=[3, 4])
    #def hvp(x, v, *argss):
    #    return grad(lambda x: jnp.vdot(func(x, *argss)[1], v))(x)

    #hvpj = jit(hvp, static_argnums=[3,4])
    hess = jit(jacfwd(jacrev(model_lossfunc)), static_argnums=[3, 4])
    opts = {'maxiter': maxiter}
   
    with tqdm(total=maxiter, delay=0.1) as tq:
        saved_params = { "iteration_number" : 0 }
        def callback(xk):
            a, b = func(xk, *arrgs)
            gNorm = np.linalg.norm(b)
            tq.set_postfix(val='{:.2e}'.format(a), g='{:.2e}'.format(gNorm), refresh=False)
            tq.update(1)
            if saved_params["iteration_number"] % 5 == 0:
                print("{:3} | {}".format(
                saved_params["iteration_number"], model_lossfunc(xk, data.values, kav, metric, lrank, 1e-9, 1e-12, nonzero_indices, r_index_matrix)))
            saved_params["iteration_number"] += 1

        print("")

        opt = minimize(func, x0, method="trust-ncg", args=arrgs, hess=hess, callback=callback, jac=True, options=opts)
        print(f"Exit message: {opt.message}")
        print(f"Exit status: {opt.status}")

    return opt.x[:, np.newaxis]

def run_optimization(data: xr.DataArray, metric, lrank=False, per_receptor=False, n_ab=1):
    data = prepare_data(data)
    kav = assemble_Kavf(data)
    kav[np.where(kav == 0)] = 10
    kav_log = np.log(kav)
    final_matrix = optimize_lossfunc(data, kav_log.values, metric, lrank=lrank, per_receptor=per_receptor, n_ab=n_ab)
    final_matrix = final_matrix[:-1]
    if lrank:
        r_subj_pred, r_ag_pred = reshape_params(final_matrix, data, lrank=lrank)
        lbound = infer_Lbound(data, kav_log.values, lrank=lrank, r_subj=r_subj_pred, r_ag=r_ag_pred)
        return r_subj_pred, r_ag_pred, lbound, kav # returns not log version of everything(!)
    else:
        r_abund_pred = reshape_params(final_matrix, data, lrank=lrank)
        lbound = infer_Lbound(data, kav_log.values, lrank=lrank, r_abund=r_abund_pred)
        return r_abund_pred, lbound, kav # returns not log version of everything (!)

if __name__ == "__main__":
    pass