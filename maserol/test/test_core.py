import pytest
from ..core import *
from ..preprocess import HIgGs, HIgGFs
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data3D as zohar

@pytest.mark.parametrize("n_ab", [1, 2, 3])
def test_initialize(n_ab):
    """ Test initializeParams() work correctly """
    cube = atyeo(xarray=True)
    cube = prepare_data(cube)
    n_samp, n_recp, n_ag = cube.shape
    ab_types = HIgGs[:n_ab]
    ps = initializeParams(cube, lrank=True, ab_types=ab_types)  # should return subj, ag
    assert len(ps) == 3
    assert ps[1].shape == (n_ag, n_ab)
    ps = initializeParams(cube, lrank=False, ab_types=ab_types)  # should return abund, Ka
    assert len(ps) == 2
    assert ps[0].shape == (n_samp * n_ag, n_ab)
    assert ps[1].shape == (n_recp, n_ab)

@pytest.mark.parametrize("ab_types", [HIgGs, HIgGFs])
def test_fit_mean(ab_types):
    """ Test mean (MSE) mode, low rank assumption, not fitting Ka """
    cube = zohar(xarray=True, logscale=False)
    cube = prepare_data(cube)
    cube.values[np.random.rand(*cube.shape) < 0.05] = np.nan    # introduce missing values
    nonneg_idx = getNonnegIdx(cube, "mean")
    R_subj_guess, R_Ag_guess, Ka = initializeParams(cube, lrank=True, ab_types=ab_types)
    x0 = flattenParams(R_subj_guess, R_Ag_guess)

    # test mean (MSE) method
    x0_loss = modelLoss(x0, cube, Ka, nonneg_idx, ab_types, metric="mean", lrank=True)
    assert x0_loss > 0.0
    assert np.isfinite(x0_loss)
    x_opt, opt_f = optimizeLoss(cube, metric="mean", lrank=True, fitKa=False, maxiter=20, ab_types=ab_types)
    assert opt_f < x0_loss
    assert len(x0) == len(x_opt) - 1  # subtract the scaling factor

def test_fit_rtot():
    """ Test Rtot mode, without low rank assumption, not fitting Ka """
    cube = zohar(xarray=True, logscale=False)
    cube = prepare_data(cube)
    cube.values[np.random.rand(*cube.shape) < 0.1] = np.nan  # introduce missing values
    Ka = assembleKav(cube).values
    Abund_guess, Ka = initializeParams(cube, lrank=False)
    x0 = flattenParams(Abund_guess)

    # test Rtot method
    x0_R2 = modelLoss(x0, cube.values, Ka, getNonnegIdx(cube, metric="rtot"), 
                        metric="rtot")
    assert np.isfinite(x0_R2)
    assert x0_R2 > -0.3
    x_opt, opt_R2 = optimizeLoss(cube, metric="rtot", lrank=False, fitKa=False, maxiter=20)
    assert opt_R2 < -0.8
    assert opt_R2 < x0_R2
    assert len(x0) == len(x_opt)


@pytest.mark.parametrize("n_ab", [2, 3])
@pytest.mark.parametrize("metric", ["rrcp", "rag"])
def test_fit_r(n_ab, metric):
    """ Test R per Receptor/Ag mode, low rank assumption, fit Ka """
    cube = zohar(xarray=True, logscale=False)
    cube = prepare_data(cube)
    cube.values[np.random.rand(*cube.shape) < 0.1] = np.nan  # introduce missing values
    ab_types = HIgGs[:n_ab]
    R_subj_guess, R_Ag_guess, Ka = initializeParams(cube, lrank=True, ab_types=ab_types)
    x0 = flattenParams(R_subj_guess, R_Ag_guess, Ka)

    # test Rtot method
    x0_R2 = modelLoss(jnp.array(x0), jnp.array(cube.values), jnp.ones_like(Ka) * -1, 
                      getNonnegIdx(cube, metric=metric), ab_types=ab_types,
                      metric=metric, lrank=True, fitKa=True)# Ka after kwargs must not be used here
    assert np.isfinite(x0_R2)
    assert x0_R2 > -0.3
    x_opt, opt_R2 = optimizeLoss(cube, metric=metric, lrank=True, fitKa=True, maxiter=20, ab_types=ab_types)
    assert opt_R2 < -0.7
    assert len(x0) == len(x_opt)