"""
Currently runs polyfc to compare with SpaceX data with 6 receptors, 117 subjects, and 14 antigens.
Polyfc is ran with initial guesses for abundance (117x1) and (14x1)  and Ka for each receptor (6x1).
Polyfc output is compared to SpaceX data and cost function is minimzied through scipy.optimize.minimize.
Total fitting parameters = 147
"""
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
from jax import value_and_grad, jvp, jit
from jax.config import config
from .model import phi_solve
from scipy.stats import pearsonr

config.update('jax_log_compiles', True)

def initial_AbundKa(cube, n_ab=1):
    """
    generate abundance and Ka matrices from random values
    cube.shape == n_subj * n_rec * n_Ag
    """
    R_subj_guess = np.random.lognormal(size=(cube.shape[0], n_ab))
    R_Ag_guess = np.random.lognormal(size=(cube.shape[2], n_ab))
    Ka_guess = np.random.lognormal(12.0, size=(cube.shape[1], n_ab))
    return R_subj_guess, R_Ag_guess, Ka_guess


def infer_Lbound(R_subj, R_Ag, Ka, L0=1e-9, KxStar=1e-12):
    """
    pass the matrices generated above into polyfc, run through each receptor
    and ant x sub pair and store in matrix same size as flatten
    """
    Phisum = jnp.zeros((R_subj.shape[0], Ka.shape[0], R_Ag.shape[0]))
    # Lbound_guess = 6x1638
    RR = jnp.einsum("ij,kj->ijk", R_subj, R_Ag)

    for jj in range(Phisum.shape[1]):
        Phisum = Phisum.at[:, jj, :].set(phi_solve(L0, KxStar, RR, Ka[jj, :, np.newaxis]))

    return L0 / KxStar * ((1.0 + Phisum) ** 2 - 1.0)


def model_lossfunc(x, cube, L0=1e-9, KxStar=1e-12):
    """
        Loss function, comparing model output and flattened tensor
    """
    # unflatten to three matrices
    n_subj, n_rec, n_Ag = cube.shape
    n_ab = int(len(x) / np.sum(cube.shape))

    R_subj = x[0:(n_subj * n_ab)].reshape(n_subj, n_ab)
    R_Ag = x[(n_subj * n_ab):((n_subj + n_Ag) * n_ab)].reshape(n_Ag, n_ab)
    Ka = x[(n_subj + n_Ag) * n_ab:(n_subj + n_Ag + n_rec) * n_ab].reshape(n_rec, n_ab)

    Lbound = infer_Lbound(R_subj, R_Ag, Ka, L0=L0, KxStar=KxStar)
    return jnp.linalg.norm(jnp.nan_to_num(Lbound - cube))


def optimize_lossfunc(cube, n_ab=1, maxiter=100):
    """
        Optimization method to minimize model_lossfunc output
    """
    R_subj_guess, R_Ag_guess, Ka_guess = initial_AbundKa(cube, n_ab=n_ab)
    x0 = np.concatenate((R_subj_guess.flatten(), R_Ag_guess.flatten(), Ka_guess.flatten()))

    func = jit(value_and_grad(model_lossfunc))
    opts = {"verbose": 1, 'maxiter': maxiter}
    bnd = [(1e-9, np.inf)] * x0.size

    def hvp(x, p, *args):
        return jvp(lambda xx: func(xx, *args)[1], (x,), (p,))[1]

    def funcc(*args):
        a, b = func(*args)
        return a, np.array(b)

    print("")
    opt = minimize(funcc, x0, method="trust-constr", args=(cube, 1e-9, 1e-12), hessp=hvp, jac=True, bounds=bnd, options=opts)
    print(opt.fun)

    return opt.x[:, np.newaxis]


def opt_to_matrices(cube, RKa_opt):
    n_subj, n_rec, n_Ag = cube.shape
    n_ab = int(len(RKa_opt) / np.sum(cube.shape))

    R_subj = RKa_opt[0:(n_subj * n_ab)].reshape(n_subj, n_ab)
    R_Ag = RKa_opt[(n_subj * n_ab):((n_subj + n_Ag) * n_ab)].reshape(n_Ag, n_ab)
    Ka = RKa_opt[(n_subj + n_Ag) * n_ab:(n_subj + n_Ag + n_rec) * n_ab].reshape(n_rec, n_ab)
    return R_subj, R_Ag, Ka


def plot_correlation_heatmap(ax, RKa_opt, cube, rec_names, ant_names):
    """
    Uses optimal parameters from optimize_lossfunc to run the model
    Generates prelim figures to compare experimental and model results
    R_subj, R_Ag, Ka, L0=L0, KxStar=KxStar
    """

    R_subj, R_Ag, Ka = opt_to_matrices(cube, RKa_opt)
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
