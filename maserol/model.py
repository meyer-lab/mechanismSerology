"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from scipy.optimize import root
from scipy.special import binom


def Req_func(Req, Rtot, L0fA, AKxStar, f):
    """ Mass balance. Transformation to account for bounds. """
    Phisum = np.dot(AKxStar, Req.T)
    return Req + L0fA * Req * (1 + Phisum) ** (f - 1) - Rtot


def polyfc(L0, KxStar, f, Rtot, LigC, Kav):
    """
    The main function. Generate all info for heterogenenous binding case
    L0: concentration of ligand complexes.
    KxStar: detailed balance-corrected Kx.
    f: valency
    Rtot: numbers of each receptor appearing on the cell.
    LigC: the composition of the mixture used.
    Kav: a matrix of Ka values. row = ligands, col = receptors
    """
    # Data consistency check
    Kav = np.array(Kav)
    Rtot = np.array(Rtot, dtype=float)
    assert Rtot.ndim <= 1
    LigC = np.array(LigC)
    assert LigC.ndim <= 1
    LigC = LigC / np.sum(LigC)
    assert LigC.size == Kav.shape[0]
    #assert Rtot.size == Kav.shape[1]
    #assert Kav.ndim == 2

    # Run least squares to get Req
    Req = Req_Regression(L0, KxStar, f, Rtot, LigC, Kav)

    nr = Rtot.size  # the number of different receptors

    Phi = np.ones((LigC.size, nr + 1)) * LigC.reshape(-1, 1)
    Phi[:, :nr] *= Kav * Req * KxStar
    Phisum = np.sum(Phi[:, :nr])

    Lbound = L0 / KxStar * ((1 + Phisum) ** f - 1)
    Rbound = L0 / KxStar * f * Phisum * (1 + Phisum) ** (f - 1)
    vieq = L0 / KxStar * binom(f, np.arange(1, f + 1)) * np.power(Phisum, np.arange(1, f + 1))

    return Lbound, Rbound, vieq


def Req_Regression(L0, KxStar, f, Rtot, LigC, Kav):
    """Run least squares regression to calculate the Req vector"""
    A = np.dot(LigC.T, Kav)
    L0fA = L0 * f * A
    AKxStar = A * KxStar

    # Identify an initial guess just on max monovalent interaction
    x0 = np.max(L0fA, axis=0)
    x0 = np.multiply(1.0 - np.divide(x0, 1 + x0), Rtot)

    # Solve Req by calling least_squares() and Req_func()
    lsq = root(Req_func, x0, method="lm", args=(Rtot, L0fA, AKxStar, f), options={"maxiter": 100000, "ftol": 1e-14})
    assert lsq["success"], "Failure in rootfinding. " + str(lsq)

    return lsq["x"].reshape(1, -1)


def Req_func2(Req, L0, KxStar, Rtot, Cplx, Ctheta, Kav):
    Psi = Req * Kav * KxStar
    Psi = np.pad(Psi, ((0, 0), (0, 1)), constant_values=1)
    Psirs = np.sum(Psi, axis=1).reshape(-1, 1)
    Psinorm = (Psi / Psirs)[:, :-1]

    Rbound = L0 / KxStar * np.sum(Ctheta.reshape(-1, 1) * np.dot(Cplx, Psinorm) * np.exp(np.dot(Cplx, np.log1p(Psirs - 1))), axis=0)
    return Req + Rbound - Rtot



def polyc(L0, KxStar, Rtot: np.ndarray, Cplx: np.ndarray, Ctheta: np.ndarray, Kav: np.ndarray):
    """
    The main function to be called for multivalent binding
    :param L0: concentration of ligand complexes
    :param KxStar: Kx for detailed balance correction
    :param Rtot: numbers of each receptor on the cell
    :param Cplx: the monomer ligand composition of each complex
    :param Ctheta: the composition of complexes
    :param Kav: Ka for monomer ligand to receptors
    :return:
        Lbound: a list of Lbound of each complex
        Rbound: a list of Rbound of each kind of receptor
    """
    # Consistency check
    Kav = np.array(Kav)
    assert Kav.ndim == 2
    Rtot = np.array(Rtot, dtype=float)
    assert Rtot.ndim == 1
    Cplx = np.array(Cplx)
    assert Cplx.ndim == 2
    Ctheta = np.array(Ctheta)
    assert Ctheta.ndim == 1

    assert Kav.shape[0] == Cplx.shape[1]
    assert Kav.shape[1] == Rtot.size
    assert Cplx.shape[0] == Ctheta.size
    Ctheta = Ctheta / np.sum(Ctheta)

    # Solve Req **?**
    lsq = root(Req_func2, Rtot, method="lm", args=(L0, KxStar, Rtot, Cplx, Ctheta, Kav), options={"maxiter": 100000, "ftol": 1e-14})
        # Root, finding roots of vector function, using LM NLLS
    assert lsq["success"], "Failure in rootfinding. " + str(lsq)
    Req = lsq["x"].reshape(1, -1)

    # Calculate the results
    Psi = np.ones((Kav.shape[0], Kav.shape[1] + 1))
    Psi[:, : Kav.shape[1]] *= Req * Kav * KxStar
    Psirs = np.sum(Psi, axis=1).reshape(-1, 1)
    Psinorm = (Psi / Psirs)[:, :-1]

    Lbound = L0 / KxStar * Ctheta * np.expm1(np.dot(Cplx, np.log(Psirs))).flatten()
    Rbound = L0 / KxStar * Ctheta.reshape(-1, 1) * np.dot(Cplx, Psinorm) * np.exp(np.dot(Cplx, np.log(Psirs)))
    Lfbnd = L0 / KxStar * Ctheta * np.exp(np.dot(Cplx, np.log(Psirs - 1.0))).flatten()
    assert len(Lbound) == len(Ctheta)
    assert Rbound.shape[0] == len(Ctheta)
    assert Rbound.shape[1] == len(Rtot)
    return Lbound, Rbound, Lfbnd