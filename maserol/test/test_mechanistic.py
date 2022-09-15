import pytest
import numpy as np
from ..core import initializeParams, model_lossfunc, optimize_lossfunc, flattenParams
from tensordata.atyeo import data as atyeo


@pytest.mark.parametrize("n_ab", [1, 2, 3])
def test_fit(n_ab):
    cube = atyeo(True)
    R_subj_guess, R_Ag_guess, Ka_guess = initializeParams(cube, n_ab)
    x0 = flattenParams(R_subj_guess, R_Ag_guess, Ka_guess)
    RKa_opt = model_lossfunc(x0, cube, metric= L0=1e-9, KxStar=1e-12)
    assert RKa_opt > 0.0
    assert np.isfinite(RKa_opt)

    optimize_lossfunc(cube, n_ab=n_ab, maxiter=500)

def test_fit_Ka():
    pass