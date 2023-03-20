import pytest
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data as zohar
from tensordata.kaplonek import MGH4D, SpaceX4D
from valentbind.model import polyc

from ..core import *
from ..preprocess import HIgGs, HIgGFs


@pytest.mark.parametrize("n_ab", [1, 2, 3])
def test_initialize(n_ab):
    """ Test initializeParams() work correctly """
    cube = atyeo()
    cube = prepare_data(cube)
    n_samp, n_recp, n_ag = cube.shape
    ab_types = HIgGs[:n_ab]
    ps = initializeParams(cube, lrank=True, ab_types=ab_types)  # should return subj, ag
    assert len(ps) == 3
    assert ps[1].shape == (n_ag, n_ab)
    ps = initializeParams(cube, lrank=False, ab_types=ab_types)  # should return abund, Ka
    assert len(ps) == 2
    assert ps[0].shape == (n_samp, n_ab, n_ag)
    assert ps[1].shape == (n_recp, n_ab)

def test_inferLbound():
    """ Test that our model here provides the same outcome as expected """
    n_subj, n_rcp, n_ag, n_ab = 6, 5, 4, 3
    FcIdx = 2
    L0 = 1e-9
    KxStar = np.random.rand() * 1e-12
    Rtot = np.random.rand(n_subj, n_ab, n_ag) * np.power(10, np.random.randint(1, 5, size=(n_subj, n_ab, n_ag)))
    Ka = np.random.rand(n_rcp, n_ab) * np.power(10, np.random.randint(5, 8, size=(n_rcp, n_ab)))

    # maserol implementation of binding model
    cube = np.zeros((n_subj, n_rcp, n_ag))
    msRes = inferLbound(cube, Rtot, Ka, lrank=False, L0=L0, KxStar=KxStar, FcIdx=FcIdx)

    # valentbind implementation of binding model
    vbRes = np.zeros((n_subj, n_rcp, n_ag))
    for i_subj in range(n_subj):
        for i_rcp in range(n_rcp):
            for i_ag in range(n_ag):
                vbRes[i_subj, i_rcp, i_ag] = polyc(
                    L0, KxStar, Rtot[i_subj, : , i_ag],
                    np.array([[4]]) if i_rcp >= FcIdx else np.array([[2]]), # f
                    np.array([1]), # Ctheta
                    Ka[[i_rcp], :])[0]

    # compare
    assert np.allclose(msRes, vbRes, rtol = 1e-4)

@pytest.mark.parametrize("ab_types", [HIgGs, HIgGFs])
def test_fit_mean(ab_types):
    """ Test mean (MSE) mode, low rank assumption, not fitting Ka """
    cube = zohar()
    cube = prepare_data(cube)
    cube.values[np.random.rand(*cube.shape) < 0.05] = np.nan    # introduce missing values
    nonneg_idx = getNonnegIdx(cube, "mean")
    R_subj_guess, R_Ag_guess, Ka = initializeParams(cube, lrank=True, ab_types=ab_types)
    x0 = flattenParams(R_subj_guess, R_Ag_guess)
    x0 = np.append(x0, 0)

    # test mean (MSE) method
    x0_loss = modelLoss(x0, cube, Ka, nonneg_idx, ab_types, metric="mean", lrank=True)
    assert x0_loss > 0.0
    assert np.isfinite(x0_loss)
    x_opt, opt_f = optimizeLoss(cube, metric="mean", lrank=True, fitKa=False, maxiter=20, ab_types=ab_types)
    assert opt_f < x0_loss
    assert len(x0) == len(x_opt)

def test_fit_rtot():
    """ Test Rtot mode, without low rank assumption, not fitting Ka """
    cube = zohar()
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
    cube = zohar()
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
    assert opt_R2 < -0.5
    assert len(x0) == len(x_opt)

@pytest.mark.parametrize("lrank", [False, True])
@pytest.mark.parametrize("data", [zohar(),
                                  atyeo(),
                                  MGH4D()["Serology"].stack(Sample = ("Subject", "Time")),
                                  SpaceX4D().stack(Sample = ("Subject", "Time"))])
def test_reshape_params(lrank, data):
    ab_types = list(HIgGs)
    cube = prepare_data(data)
    if lrank:
        sample, ag, Ka = initializeParams(cube, lrank=lrank, ab_types=ab_types)
        x = flattenParams(sample, ag, Ka)
        sample, ag, Ka = reshapeParams(x, cube, lrank=lrank, fitKa=True, as_xarray=True, ab_types=ab_types)
        assert (sample.Sample.values == cube.Sample.values).all()
        assert (sample.Antibody.values == ab_types).all()
        assert (ag.Antigen.values == cube.Antigen.values).all()
        assert (ag.Antibody.values == ab_types).all()
    else:
        abundance, Ka = initializeParams(cube, lrank=lrank, ab_types=ab_types)
        x = flattenParams(abundance, Ka)
        abundance, Ka = reshapeParams(x, cube, lrank=lrank, fitKa=True, as_xarray=True, ab_types=ab_types)
        assert (abundance.Sample.values == cube.Sample.values).all()
        assert (abundance.Antigen.values == cube.Antigen.values).all()
        assert (abundance.Antibody.values == ab_types).all()

def test_factor_abundance():
    n_comps = 1
    n_sample = 400
    n_ag = 6
    n_ab = 8
    scaling_factor = 2000
    sample_facs = np.random.rand(n_sample, n_ab, n_comps) * scaling_factor
    ag_facs = np.random.rand(n_ag, n_ab, n_comps)
    abundance = reconstructAbundance(sample_facs, ag_facs)
    abundance_xr = xr.DataArray(
        abundance,
        dims=("Sample", "Antibody", "Antigen"),
        coords=(np.arange(n_sample), np.arange(n_ab), np.arange(n_ag))
    )
    got_sample_facs, got_ag_facs = factorAbundance(abundance_xr, n_comps, as_xarray=False)

    def normalized_error(x1, x2): 
        return np.linalg.norm(x1 - x2) / np.linalg.norm(x2)

    want_sample_mean = np.mean(sample_facs)
    sample_mean_mult = want_sample_mean / np.mean(got_sample_facs)
    got_sample_facs = got_sample_facs * sample_mean_mult
    got_ag_facs = got_ag_facs / sample_mean_mult

    baseline = np.random.rand(*sample_facs.shape)
    baseline = baseline * want_sample_mean / np.mean(baseline)

    # assert that we matched our scales correctly
    assert np.isclose(np.mean(baseline), np.mean(got_sample_facs))
    assert np.isclose(np.mean(got_sample_facs), np.mean(sample_facs))

    # assert that the factors we got from factorAbundance are somewhat close to the factors we used to construct our original tensor
    assert normalized_error(got_sample_facs, sample_facs) < 0.6 * normalized_error(baseline, sample_facs)

    got_abundance = reconstructAbundance(got_sample_facs, got_ag_facs)
    baseline = np.random.rand(*abundance.shape) * np.mean(abundance)
    # assert that the reconstruction error is small
    assert normalized_error(got_abundance, abundance) < 0.01 * normalized_error(baseline, abundance)
