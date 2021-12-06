"""
Currently runs polyfc to compare with SpaceX data with 6 receptors, 117 subjects, and 14 antigens.
Polyfc is ran with initial guesses for abundance (117x1) and (14x1)  and Ka for each receptor (6x1).
Polyfc output is compared to SpaceX data and cost function is minimzied through scipy.optimize.minimize.
Total fitting parameters = 147
"""
import numpy as np
import jax.numpy as jnp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from jax import jacrev
from .model import lBnd


def initial_AbundKa(cube, n_ab=1):
    """
    generate abundance and Ka matrices from random values
    cube.shape == n_subj * n_rec * n_Ag
    """
    R_subj_guess = np.random.lognormal(size=(cube.shape[0], n_ab))
    R_Ag_guess = np.random.lognormal(size=(cube.shape[2], n_ab))
    Ka_guess = np.random.lognormal(6, size=(cube.shape[1], n_ab))
    return R_subj_guess, R_Ag_guess, np.log(Ka_guess)


def infer_Lbound(R_subj, R_Ag, Ka, L0=1e-9, KxStar=1e-12):
    """
    pass the matrices generated above into polyfc, run through each receptor
    and ant x sub pair and store in matrix same size as flatten
    """
    Lbound_cube = jnp.zeros((R_subj.shape[0], Ka.shape[0], R_Ag.shape[0]))
    # Lbound_guess = 6x1638
    Ka = jnp.exp(Ka[:, np.newaxis])
    RR = jnp.einsum("ij,kj->ijk", R_subj, R_Ag)

    for jj in range(Lbound_cube.shape[1]):
        Lbound_cube = Lbound_cube.at[:, jj, :].set(lBnd(L0, KxStar, RR, Ka[jj, :]))

    return Lbound_cube


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
    return (Lbound - cube).flatten()


def optimize_lossfunc(cube, n_ab=1, maxiter=100):
    """
        Optimization method to minimize model_lossfunc output
    """
    R_subj_guess, R_Ag_guess, Ka_guess = initial_AbundKa(cube, n_ab=n_ab)
    x0 = np.concatenate((R_subj_guess.flatten(), R_Ag_guess.flatten(), Ka_guess.flatten()))

    opt = least_squares(model_lossfunc, x0, args=(cube, 1e-9, 1e-12), jac=jacrev(model_lossfunc), bounds=(0, np.inf), verbose=2, max_nfev=maxiter)

    RKa_opt = opt.x[:, np.newaxis]
    return RKa_opt


def compare(RKa_opt, cube):
    """
    Uses optimal parameters from optimize_lossfunc to run the model
    Generates prelim figures to compare experimental and model results
    """
    Lbound_model = infer_Lbound(RKa_opt[:cube.shape[1], :], RKa_opt[cube.shape[1]:, :])

    for ii in range(cube.shape[0]):
        plt.plot(cube[ii, :], Lbound_model[ii, :])
