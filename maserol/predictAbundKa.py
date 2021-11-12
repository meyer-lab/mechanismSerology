"""
Currently runs polyfc to compare with SpaceX data with 6 receptors, 117 subjects, and 14 antigens.
Polyfc is ran with initial guesses for abundance (117x1) and (14x1)  and Ka for each receptor (6x1).
Polyfc output is compared to SpaceX data and cost function is minimzied through scipy.optimize.minimize.
Total fitting parameters = 147
"""
import numpy as np
from scipy.sparse import csr_matrix
from jax import jacrev, jit
import jax.numpy as jnp
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt


def Req_func(Req: np.ndarray, Rtot: np.ndarray, L0: float, KxStar: float, Kav: np.ndarray):
    """ Mass balance. """
    L0fA = L0 * 2 * Kav
    AKxStar = Kav * KxStar
    Phisum = jnp.dot(AKxStar, Req.T)
    term = jnp.einsum("ij,kil,ilk->kil", L0fA, Req, 1 + Phisum)
    return Req + term - Rtot


def lBnd(L0: float, KxStar, Rtot, Kav):
    """
    The main function. Generate all info for heterogenenous binding case
    L0: concentration of ligand complexes.
    KxStar: detailed balance-corrected Kx.
    Rtot: numbers of each receptor appearing on the cell.
    Kav: a matrix of Ka values. row = ligands, col = receptors
    """
    # Run least squares to get Req
    def bal(x):
        xR = np.reshape(x, Rtot.shape)
        return Req_func(xR, Rtot, L0, KxStar, Kav).flatten()

    jacc = jit(jacrev(bal))

    def jaccFunc(x):
        J = csr_matrix(jacc(x))
        J.eliminate_zeros()
        return J

    x0 = Rtot.flatten() / 1000.0
    bnd = (0.0, Rtot.flatten())
    lsq = least_squares(bal, x0, jac=jaccFunc, bounds=bnd, xtol=1e-9, tr_solver="lsmr")
    assert lsq.success, "Failure in rootfinding. " + str(lsq)

    Req = np.reshape(lsq.x, Rtot.shape)

    AKxStar = Kav * KxStar
    Phisum = np.dot(AKxStar, Req.T)

    Lbound = L0 / KxStar * ((1 + Phisum) ** 2 - 1)
    return np.squeeze(Lbound).T


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
    Ka = np.exp(Ka[:, np.newaxis])
    RR = np.einsum("ij,kj->ijk", R_subj, R_Ag)

    for jj in range(Lbound_cube.shape[1]):
        Lbound_cube[:, jj, :] = lBnd(L0, KxStar, RR, Ka[jj, :])

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
