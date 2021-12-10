""" Import binding affinities. """

from os.path import join, dirname
import numpy as np
import pandas as pd
from jax.config import config
import jax.numpy as jnp
from jaxopt import ScipyLeastSquares, FixedPointIteration


path_here = dirname(dirname(__file__))


config.update("jax_enable_x64", True)


def phi(Phisum, Rtot, L0, KxStar, Kav):
    Req = Rtot / (1.0 + 2.0 * L0 * Kav * (1.0 + Phisum))
    Phisum = jnp.dot(Kav * KxStar, Req.T)
    return Phisum


def lBnd(L0: float, KxStar, Rtot, Kav):
    """
    The main function. Generate all info for heterogenenous binding case
    L0: concentration of ligand complexes.
    KxStar: detailed balance-corrected Kx.
    Rtot: numbers of each receptor appearing on the cell.
    Kav: a matrix of Ka values. row = ligands, col = receptors
    """
    fpi = FixedPointIteration(fixed_point_fun=phi, tol=1e-12, implicit_diff=True)
    fpout = fpi.run(0.0, Rtot, L0, KxStar, Kav)
    Phisum = np.squeeze(fpout.params)
    Lbound = L0 / KxStar * ((1.0 + Phisum) ** 2 - 1.0)
    assert False
    return Lbound


def human_affinity():
    return pd.read_csv(join(path_here, "maserol/data/murine-affinities.csv"),
                       delimiter=",", comment="#", index_col=0)
