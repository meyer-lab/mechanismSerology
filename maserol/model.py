""" Import binding affinities. """

from os.path import join, dirname
import numpy as np
import pandas as pd
from jax.config import config
import jax.numpy as jnp
from jaxopt import ScipyLeastSquares


path_here = dirname(dirname(__file__))


config.update("jax_enable_x64", True)


def Req_func(Req: np.ndarray, Rtot: np.ndarray, L0fA: np.ndarray, AKxStar: np.ndarray):
    """ Mass balance. """
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
    L0fA = L0 * 2 * Kav
    AKxStar = Kav * KxStar

    x0 = Rtot.flatten()

    # Run least squares to get Req
    def bal(x, *args):
        xR = jnp.reshape(x, Rtot.shape)
        return Req_func(xR, *args).flatten()
    
    x0 = x0 / (1.0 + L0fA) # Monovalent guess

    lsq = ScipyLeastSquares('trf', fun=bal, options={"xtol": 1e-9, "gtol": 1e-12})
    lsq = lsq.run(x0, Rtot, L0fA, AKxStar)
    assert lsq.state.success, "Failure in rootfinding. " + str(lsq)
    assert lsq.state.cost_val < 1.0e-9
    Req = jnp.reshape(lsq.params, Rtot.shape)

    Phisum = jnp.dot(AKxStar, Req.T)
    Lbound = L0 / KxStar * ((1 + Phisum) ** 2 - 1)
    return jnp.squeeze(Lbound).T


def human_affinity():
    return pd.read_csv(join(path_here, "maserol/data/murine-affinities.csv"),
                       delimiter=",", comment="#", index_col=0)
