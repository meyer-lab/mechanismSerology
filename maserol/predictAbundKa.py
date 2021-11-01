import numpy as np
from valentbind import polyfc

def initial_AbundKa(flatCube, n_ab=1):
    """
    generate abundance and Ka matrices from random values
    flatCube.shape == n_rec * (n_subj * n_Ag)
    """
    R_guess = np.random.randint(4, high=11, size=(n_ab, flatCube.shape[1]), dtype=int)
    Ka_guess = np.random.randint(1e8, high=9e8, size=(flatCube.shape[0], n_ab), dtype=int)
    return R_guess, Ka_guess


def infer_Lbound(R, Ka, L0=1e-9, KxStar=1e-12):
    """
    pass the matrices generated above into polyfc, run through each receptor
    and ant x sub pair and store in matrix same size as flatten
    """
    Lbound_guess = np.zeros((Ka.shape[0], R.shape[1]))
    LigC = np.array([1])
    for ii in range(Lbound_guess.shape[0]):
        for xx in range(Lbound_guess.shape[1]):
            Lbound_guess[ii,xx] = polyfc(L0, KxStar, 2, R[:, xx], LigC, Ka[[ii], :])[0]
    return Lbound_guess


def model_lossfunc(R, Ka, flatCube):
    # loss function, comparing model output and flattened tensor
    Lbound_guess = infer_Lbound(R, Ka)
    model_loss = np.nansum((Lbound_guess - flatCube)**2)
    return model_loss
