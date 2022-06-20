import pytest
import numpy as np
from ..predictAbundKa import initial_AbundKa, model_lossfunc, optimize_lossfunc
from tensordata.atyeo import createCube


@pytest.mark.parametrize("n_ab", [1, 2, 3])
def test_fit(n_ab):
    cube = createCube()
    R_subj_guess, R_Ag_guess, Ka_guess = initial_AbundKa(cube, n_ab)
    x0 = np.concatenate((R_subj_guess.flatten(), R_Ag_guess.flatten(), Ka_guess.flatten()))
    RKa_opt = model_lossfunc(x0, cube, 1e-9, 1e-12)
    assert RKa_opt > 0.0
    assert np.isfinite(RKa_opt)

    optimize_lossfunc(cube, n_ab=n_ab, maxiter=2000)
