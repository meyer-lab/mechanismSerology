import pytest
from ..mechanistic import *
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data3D as zohar

@pytest.mark.parametrize("n_ab", [1, 2, 3])
def test_initialize(n_ab):
    """ Test initializeParams() work correctly """
    cube = atyeo(xarray=True)
    n_samp, n_recp, n_ag = cube.shape
    ps = initializeParams(cube, lrank=True, fitKa=False, n_ab=n_ab)  # should return subj, ag
    assert len(ps) == 2
    assert ps[1].shape == (n_ag, n_ab)
    ps = initializeParams(cube, lrank=False, fitKa=True, n_ab=n_ab)  # should return abund, Ka
    assert len(ps) == 2
    assert ps[0].shape == (n_samp * n_ag, n_ab)
    assert ps[1].shape == (n_recp, n_ab)

@pytest.mark.parametrize("fucose", [True, False])
def test_fit_mean(fucose):
    cube = zohar(xarray=True, logscale=False)
    cube = prepare_data(cube)
    Ka = assemble_Kav(cube, fucose=fucose).values
    R_subj_guess, R_Ag_guess = initializeParams(cube, lrank=True, fitKa=False, n_ab=Ka.shape[1])
    x0 = flattenParams(R_subj_guess, R_Ag_guess)

    # test mean (MSE) method
    RKa_opt = model_lossfunc(x0, cube,
                             "mean", True, False, 1e-9, 1e-12, Ka)  # = metric, lrank, fitKa, L0, KxStar, Ka
    assert RKa_opt > 0.0
    assert np.isfinite(RKa_opt)
    x_opt, opt_f = optimize_lossfunc(cube, metric="mean", lrank=True, fitKa=False, maxiter=20, fucose=fucose)
    assert opt_f < RKa_opt
    assert len(x0) == len(x_opt) - 1  # subtract the scaling factor

def test_fit_rtot():
    pass

def test_fit_r():
    pass

