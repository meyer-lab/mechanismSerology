import numpy as np
from valentbind import polyfc
from scipy.optimize import least_squares, minimize
import scipy as scipy
import matplotlib.pyplot as plt

"""
Currently runs polyfc to compare with SpaceX data with 6 receptors, 117 subjects, and 14 antigens.
Polyfc is ran with initial guesses for abundance (1638x1) and Ka for each receptor (6x1).
Polyfc output is compared to SpaceX data and cost function is minimzied through NLLS.
Total fitting parameters = 1644
Runs polyfc 9,828 times

"""

def initial_AbundKa(flatCube, n_ab=1):
    """
    generate abundance and Ka matrices from random values
    flatCube.shape == n_rec * (n_subj * n_Ag)
    """
    R_guess = np.random.randint(4, high=11, size=(n_ab, flatCube.shape[1]), dtype=int)
    Ka_guess = np.random.randint(1e6, high=9e6, size=(flatCube.shape[0], n_ab), dtype=int)
    RKa_combined = np.concatenate((R_guess.T,Ka_guess))
    return RKa_combined


def infer_Lbound(R, Ka, L0=1e-9, KxStar=1e-12):
    """
    pass the matrices generated above into polyfc, run through each receptor
    and ant x sub pair and store in matrix same size as flatten
    """

    Lbound_guess = np.zeros((Ka.shape[0], R.shape[0]))
    #Lbound_guess = 6x1638
    LigC = np.array([1])
    Ka = Ka[:,np.newaxis]
    for ii in range(Lbound_guess.shape[0]):
        for xx in range(Lbound_guess.shape[1]):
            Lbound_guess[ii,xx] = polyfc(L0, KxStar, 2, R[xx,:], LigC, Ka[ii,:])[0]
    return Lbound_guess


def model_lossfunc(RKa_temp):
    """
        Loss function, comparing model output and flattened tensor
    """
    flatCube, _, _ = flattenSpaceX()
    RKa_temp = RKa_temp[:,np.newaxis]
    Lbound_guess = infer_Lbound(RKa_temp[:flatCube.shape[1],:],RKa_temp[flatCube.shape[1]:,:])
    model_loss = np.nansum((Lbound_guess - flatCube)**2)
    print(model_loss)
    return model_loss

def optimize_lossfunc(RKa,flatCube):
    """
        Optimization method to minimize model_lossfunc output
    """
    ls_sol = least_squares(model_lossfunc,RKa[:,0])
    RKa_opt = ls_sol.x
    RKa_opt = RKa_opt[:,np.newaxis]
    return RKa_opt

def compare(Rka_opt, flatCube):
    """
        Uses optimal parameters from optimize_lossfunc to run the model
        Generates prelim figures to compare experimental and model results
    """
    Lbound_model = infer_Lbound(RKa_opt[:flatCube.shape[1],:], RKa_opt[flatCube.shape[1]:,:])
    for ii in range(flatCube.shape[0]):
        plt.plot(flatCube[ii,:],Lbound_model[ii,:]);
