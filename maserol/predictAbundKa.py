"""
Currently runs polyfc to compare with SpaceX data with 6 receptors, 117 subjects, and 14 antigens.
Polyfc is ran with initial guesses for abundance (117x1) and (14x1)  and Ka for each receptor (6x1).
Polyfc output is compared to SpaceX data and cost function is minimzied through scipy.optimize.minimize.
Total fitting parameters = 147
"""
import numpy as np
from valentbind import polyfc
from .data.kaplonek import cubeSpaceX
from scipy.optimize import least_squares, minimize
from tensorly.tenalg import khatri_rao
import matplotlib.pyplot as plt


def initial_AbundKa(cube, n_ab=1):
    """
    generate abundance and Ka matrices from random values
    cube.shape == n_subj * n_rec * n_Ag
    """
    R_subj_guess = np.random.randint(1, high=5, size=(cube.shape[0], n_ab), dtype=int)
    R_Ag_guess = np.random.randint(1, high=5, size=(cube.shape[2], n_ab), dtype=int)
    Ka_guess = np.random.randint(1e6, high=9e6, size=(cube.shape[1], n_ab), dtype=int)
    return R_subj_guess, R_Ag_guess, np.log(Ka_guess)


def infer_Lbound(R_subj, R_Ag, Ka, L0=1e-9, KxStar=1e-12):
    """
    pass the matrices generated above into polyfc, run through each receptor
    and ant x sub pair and store in matrix same size as flatten
    """
    Lbound_cube = np.zeros((R_subj.shape[0], Ka.shape[0], R_Ag.shape[0]))
    # Lbound_guess = 6x1638
    LigC = np.array([1])
    Ka = Ka[:, np.newaxis]
    for jj in range(Lbound_cube.shape[1]):
        for ii in range(Lbound_cube.shape[0]):
            for kk in range(Lbound_cube.shape[2]):
                Lbound_cube[ii, jj, kk] = polyfc(L0, KxStar, 2, R_subj[ii, :] * R_Ag[kk, :], LigC, np.exp(Ka[jj, :]))[0]
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
    model_loss = np.nansum((Lbound - cube)**2)
    print("Model loss: ", model_loss)
    return model_loss


def optimize_lossfunc(cube, n_ab=1):
    """
        Optimization method to minimize model_lossfunc output
    """

    R_subj_guess, R_Ag_guess, Ka_guess = initial_AbundKa(cube, n_ab=n_ab)
    x0 = np.concatenate((R_subj_guess.flatten(), R_Ag_guess.flatten(), Ka_guess.flatten()))
    bnds = ((0, np.inf), ) * len(x0)

    opt = minimize(model_lossfunc, x0, args=(cube, 1e-9, 1e-12), bounds=bnds, options={"maxiter": 10})

    RKa_opt = opt.x
    RKa_opt = RKa_opt[:, np.newaxis]
    return RKa_opt


def compare(Rka_opt, cube):
    """
    Uses optimal parameters from optimize_lossfunc to run the model
    Generates prelim figures to compare experimental and model results
    """
    Lbound_model = infer_Lbound(RKa_opt[:cube.shape[1], :], RKa_opt[cube.shape[1]:, :])

    for ii in range(cube.shape[0]):
        plt.plot(cube[ii, :], Lbound_model[ii, :])
