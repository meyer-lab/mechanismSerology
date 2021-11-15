import numpy as np
from ..predictAbundKa import initial_AbundKa, model_lossfunc
from ..data.kaplonek import cubeSpaceX


def test_SpaceX():
    cube = cubeSpaceX()
    R_subj_guess, R_Ag_guess, Ka_guess = initial_AbundKa(cube, 1)
    x0 = np.concatenate((R_subj_guess.flatten(), R_Ag_guess.flatten(), Ka_guess.flatten()))
    RKa_opt = model_lossfunc(x0, cube, 1e-9, 1e-12)
    assert RKa_opt > 0.0
