"""
Currently runs polyfc to compare with SpaceX data with 6 receptors, 117 subjects, and 14 antigens.
Polyfc is ran with initial guesses for abundance (117x1) and (14x1)  and Ka for each receptor (6x1).
Polyfc output is compared to SpaceX data and cost function is minimzied through scipy.optimize.minimize.
Total fitting parameters = 147
"""
import numpy as np
from valentbind import polyfc
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def Req_func(Req, Rtot: np.ndarray, L0fA, AKxStar):
    """ Mass balance. """
    pPhisum = 1 + np.dot(AKxStar, Req.T)
    one = L0fA * Req
    # j and l might need to be swapped in second term
    two = np.einsum("ijkl,jmlki->imkj", one, pPhisum)
    return Req + two - Rtot[:, :, :, np.newaxis]


def Lbnd(L0: float, KxStar, Rtot, Kav):
    """
    The main function.
    """
    L0fA = L0 * 2.0 * Kav
    AKxStar = Kav * KxStar

    # TODO: Figure out the shape of Req
    # Req should have one extra dimension compared to Rtot, because Rtot is shared across detections
    Req = np.zeros((Rtot.shape[0], Rtot.shape[1], Rtot.shape[2], Kav.shape[0]))

    # TODO: Just get sizes to match, then figure out solver

    balance = Req_func(Req, Rtot, L0fA, AKxStar)
    assert balance.size == Req.size

    Phisum = np.dot(AKxStar, Req.T)
    print(Phisum.shape)
    return L0 / KxStar * (np.square(1 + Phisum) - 1)


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
    Lbound_cube = np.zeros((R_subj.shape[0], Ka.shape[0], R_Ag.shape[0]))
    # Lbound_guess = 6x1638
    LigC = np.array([1])
    Ka = np.exp(Ka[:, np.newaxis])
    RR = np.einsum("ij,kj->ijk", R_subj, R_Ag)

    lOut = Lbnd(L0, KxStar, RR, Ka)
    print(lOut.shape)
    print(Lbound_cube.shape)
    assert lOut.shape == Lbound_cube.shape

    it = np.nditer(Lbound_cube, flags=['multi_index'])
    for _ in it:
        ii, jj, kk = it.multi_index
        Lbound_cube[ii, jj, kk] = polyfc(L0, KxStar, 2, RR[ii, :, kk], LigC, Ka[jj, :])[0]

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


def optimize_lossfunc(cube, n_ab=1, maxiter=100):
    """
        Optimization method to minimize model_lossfunc output
    """
    R_subj_guess, R_Ag_guess, Ka_guess = initial_AbundKa(cube, n_ab=n_ab)
    x0 = np.concatenate((R_subj_guess.flatten(), R_Ag_guess.flatten(), Ka_guess.flatten()))
    bnds = ((0, np.inf), ) * len(x0)

    opt = minimize(model_lossfunc, x0, args=(cube, 1e-9, 1e-12), bounds=bnds, options={"maxiter": maxiter})

    RKa_opt = opt.x[:, np.newaxis]
    return RKa_opt


def compare(Rka_opt, cube):
    """
    Uses optimal parameters from optimize_lossfunc to run the model
    Generates prelim figures to compare experimental and model results
    """
    Lbound_model = infer_Lbound(RKa_opt[:cube.shape[1], :], RKa_opt[cube.shape[1]:, :])

    for ii in range(cube.shape[0]):
        plt.plot(cube[ii, :], Lbound_model[ii, :])
