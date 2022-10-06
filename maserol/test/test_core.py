import pytest
from ..core import *
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
    """ Test mean (MSE) mode, low rank assumption, not fitting Ka """
    cube = zohar(xarray=True, logscale=False)
    cube = prepare_data(cube)
    cube.values[np.random.rand(*cube.shape) < 0.05] = np.nan    # introduce missing values
    Ka = assemble_Kav(cube, fucose=fucose).values
    R_subj_guess, R_Ag_guess = initializeParams(cube, lrank=True, fitKa=False, n_ab=Ka.shape[1])
    x0 = flattenParams(R_subj_guess, R_Ag_guess)

    # test mean (MSE) method
    x0_loss = modelLoss(x0, cube,
                             "mean", True, False, 1e-9, 1e-12, Ka)  # = metric, lrank, fitKa, L0, KxStar, Ka
    assert x0_loss > 0.0
    assert np.isfinite(x0_loss)
    x_opt, opt_f = optimizeLoss(cube, metric="mean", lrank=True, fitKa=False, maxiter=20, fucose=fucose)
    assert opt_f < x0_loss
    assert len(x0) == len(x_opt) - 1  # subtract the scaling factor

def test_fit_rtot():
    """ Test Rtot mode, without low rank assumption, not fitting Ka """
    cube = zohar(xarray=True, logscale=False)
    cube = prepare_data(cube)
    cube.values[np.random.rand(*cube.shape) < 0.1] = np.nan  # introduce missing values
    Ka = assemble_Kav(cube, fucose=False).values
    Abund_guess = initializeParams(cube, lrank=False, fitKa=False, n_ab=Ka.shape[1])
    assert len(Abund_guess) == 1
    x0 = flattenParams(Abund_guess[0])

    # test Rtot method
    x0_R2 = modelLoss(x0, cube.values,
                        "rtot", False, False, 1e-9, 1e-12,   # = metric, lrank, fitKa, L0, KxStar
                        Ka, getNonnegIdx(cube, metric="rtot"))
    assert np.isfinite(x0_R2)
    assert x0_R2 > -0.3
    x_opt, opt_R2 = optimizeLoss(cube, metric="rtot", lrank=False, fitKa=False, maxiter=20, fucose=False)
    assert opt_R2 < -0.8
    assert len(x0) == len(x_opt)


@pytest.mark.parametrize("n_ab", [2, 3])
@pytest.mark.parametrize("metric", ["rrcp", "rag"])
def test_fit_r(n_ab, metric):
    """ Test R per Receptor/Ag mode, low rank assumption, fit Ka """
    cube = zohar(xarray=True, logscale=False)
    cube = prepare_data(cube)
    cube.values[np.random.rand(*cube.shape) < 0.1] = np.nan  # introduce missing values
    R_subj_guess, R_Ag_guess, Ka_guess = initializeParams(cube, lrank=True, fitKa=True, n_ab=n_ab)
    x0 = flattenParams(R_subj_guess, R_Ag_guess, Ka_guess)

    # test Rtot method
    x0_R2 = modelLoss(jnp.array(x0), jnp.array(cube.values),
                        metric, True, True, 1e-9, 1e-12,   # = metric, lrank, fitKa, L0, KxStar
                        jnp.ones_like(Ka_guess) * -1, getNonnegIdx(cube, metric=metric))
                        # Ka after kwargs must not be used here
    assert np.isfinite(x0_R2)
    assert x0_R2 > -0.3
    x_opt, opt_R2 = optimizeLoss(cube, metric=metric, lrank=True, fitKa=True, n_ab=n_ab, maxiter=20, fucose=False)
    assert opt_R2 < -0.7
    assert len(x0) == len(x_opt)