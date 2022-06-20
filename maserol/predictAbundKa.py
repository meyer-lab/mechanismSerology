"""
Currently runs polyfc to compare with SpaceX data with 6 receptors, 117 subjects, and 14 antigens.
Polyfc is ran with initial guesses for abundance (117x1) and (14x1)  and Ka for each receptor (6x1).
Polyfc output is compared to SpaceX data and cost function is minimzied through scipy.optimize.minimize.
Total fitting parameters = 147
"""
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from scipy.optimize import minimize
from jax import value_and_grad, jvp, jit
from jax.config import config
from scipy.stats import pearsonr
from tensorly.decomposition import non_negative_parafac


config.update("jax_enable_x64", True)


def initial_AbundKa(cube, n_ab=1):
    """
    generate abundance and Ka matrices from linear analysis
    cube.shape == n_subj * n_rec * n_Ag
    """
    # TODO: Add masking
    outt = non_negative_parafac(np.nan_to_num(cube), rank=n_ab)
    return outt.factors


def phi(Phisum, Rtot, L0, KxStar, Kav):
    temp = jnp.einsum("jl,ijk->ilkj", Kav, 1.0 + Phisum)
    Req = Rtot[:, :, :, np.newaxis] / (1.0 + 2.0 * L0 * temp)
    Phisum_n = jnp.einsum("jl,ilkj->ijk", Kav * KxStar, Req)
    assert Phisum_n.shape == Phisum.shape
    return Phisum_n


def infer_Lbound(R_subj, R_Ag, Ka, L0=1e-9, KxStar=1e-12):
    """
    pass the matrices generated above into polyfc, run through each receptor
    and ant x sub pair and store in matrix same size as flatten
    """
    Phisum = jnp.zeros((R_subj.shape[0], Ka.shape[0], R_Ag.shape[0]))
    Rtot = jnp.einsum("ij,kj->ijk", R_subj, R_Ag)

    for ii in range(5):
        Phisum_n = phi(Phisum, Rtot, L0, KxStar, Ka)
        Phisum = Phisum_n

    return L0 / KxStar * ((1.0 + Phisum) ** 2 - 1.0)


def reshapeParams(x, cube):
    # unflatten to three matrices
    x = jnp.exp(x)
    n_subj, n_rec, n_Ag = cube.shape
    n_ab = int(len(x) / np.sum(cube.shape))

    R_subj = x[0:(n_subj * n_ab)].reshape(n_subj, n_ab)
    R_Ag = x[(n_subj * n_ab):((n_subj + n_Ag) * n_ab)].reshape(n_Ag, n_ab)
    Ka = x[(n_subj + n_Ag) * n_ab:(n_subj + n_Ag + n_rec) * n_ab].reshape(n_rec, n_ab)
    return R_subj, R_Ag, Ka


def flattenParams(R_subj, R_Ag, Ka):
    """ Flatten into a parameter vector. """
    return np.log(np.concatenate((R_subj.flatten(), R_Ag.flatten(), Ka.flatten())))


def model_lossfunc(x, cube, L0=1e-9, KxStar=1e-12):
    """
        Loss function, comparing model output and flattened tensor
    """
    R_subj, R_Ag, Ka = reshapeParams(x, cube)
    Lbound = infer_Lbound(R_subj, R_Ag, Ka, L0=L0, KxStar=KxStar)

    # Scaling factor
    diff = jnp.log(cube) - jnp.log(Lbound)
    diff -= jnp.nanmean(diff)
    return jnp.linalg.norm(jnp.nan_to_num(diff))


def optimize_lossfunc(cube, n_ab=1, maxiter=100):
    """
        Optimization method to minimize model_lossfunc output
    """
    R_subj_guess, R_Ag_guess, Ka_guess = initial_AbundKa(cube, n_ab=n_ab)
    x0 = flattenParams(R_subj_guess, R_Ag_guess, Ka_guess)

    func = jit(value_and_grad(model_lossfunc))
    opts = {'maxiter': maxiter}
    arrgs = (cube, 1e-9, 1e-12)

    def hvp(x, p, *args):
        return jvp(lambda xx: func(xx, *args)[1], (x,), (p,))[1]

    with tqdm(total=maxiter, delay=0.1) as tq:
        def callback(xk):
            a, b = func(xk, *arrgs)
            gNorm = np.linalg.norm(b)
            tq.set_postfix(val='{:.2e}'.format(a), g='{:.2e}'.format(gNorm), refresh=False)
            tq.update(1)

        print("")
        opt = minimize(func, x0, method="trust-ncg", args=arrgs, hessp=hvp, callback=callback, jac=True, options=opts)

    return opt.x[:, np.newaxis]


def plot_correlation_heatmap(ax, RKa_opt, cube, rec_names, ant_names):
    """
    Uses optimal parameters from optimize_lossfunc to run the model
    Generates prelim figures to compare experimental and model results
    R_subj, R_Ag, Ka, L0=L0, KxStar=KxStar
    """

    R_subj, R_Ag, Ka = reshapeParams(RKa_opt, cube)
    Lbound_model = infer_Lbound(R_subj, R_Ag, Ka, L0=1e-9, KxStar=1e-12)

    coeff = np.zeros([cube.shape[1], cube.shape[2]])
    for ii in range(cube.shape[1]):
        for jj in range(cube.shape[2]):
            coeff[ii, jj], _ = pearsonr(cube[:, ii, jj], Lbound_model[:, ii, jj])

    print(coeff)
    ax.imshow(coeff)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(ant_names)))
    ax.set_xticklabels(ant_names, rotation=45)
    ax.set_yticks(np.arange(len(rec_names)))
    ax.set_yticklabels(rec_names)

    # Loop over data dimensions and create text annotations.
    for i in range(len(ant_names)):
        for j in range(len(rec_names)):
            text = ax.text(i, j, round(coeff[j, i], 2), ha="center", va="center", color="w")
