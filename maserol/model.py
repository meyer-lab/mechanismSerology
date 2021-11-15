""" Import binding affinities. """

from os.path import join, dirname
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from jax import jacrev, jit
from jax.config import config
import jax.numpy as jnp
from scipy.optimize import least_squares


path_here = dirname(dirname(__file__))


config.update("jax_enable_x64", True)


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

    x0 = Rtot.flatten()
    bnd = (0.0, Rtot.flatten())
    x0 = x0 / (1.0 + 2.0 * L0 * np.amax(Kav)) # Monovalent guess using highest affinity
    lsq = least_squares(bal, x0, jac=jaccFunc, bounds=bnd, xtol=1e-9, tr_solver="lsmr")
    assert lsq.success, "Failure in rootfinding. " + str(lsq)

    Req = np.reshape(lsq.x, Rtot.shape)

    AKxStar = Kav * KxStar
    Phisum = np.dot(AKxStar, Req.T)

    Lbound = L0 / KxStar * ((1 + Phisum) ** 2 - 1)
    return np.squeeze(Lbound).T


def human_affinity():
    return pd.read_csv(join(path_here, "maserol/data/murine-affinities.csv"),
                       delimiter=",", comment="#", index_col=0)
