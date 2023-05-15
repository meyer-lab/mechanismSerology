import pytest
import numpy as np
import xarray as xr
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data as zohar
from tensordata.kaplonek import SpaceX4D
from valentbind.model import polyc

from ..core import initializeParams, inferLbound, assembleKav, flattenParams, reconstructAbundance, reshapeParams, factorAbundance, optimizeLoss
from ..preprocess import HIgGs, HIgGFs, prepare_data


@pytest.mark.parametrize("n_ab", [1, 2, 3])
def test_initialize(n_ab):
    """ Test initializeParams() work correctly """
    cube = atyeo()
    cube = prepare_data(cube)
    n_samp, n_recp, n_ag = cube.shape
    ab_types = HIgGs[:n_ab]
    ps = initializeParams(cube, ab_types=ab_types)  # should return abund, Ka
    assert len(ps) == 2
    assert ps[0].shape == (n_samp, n_ab, n_ag)
    assert ps[1].shape == (n_recp, n_ab)


def test_inferLbound_matches_valentbind():
    """ Test that our model here provides the same outcome as expected """
    n_subj, n_rcp, n_ag, n_ab = 6, 5, 4, 3
    FcIdx = 2
    L0 = 1e-9
    KxStar = np.random.rand() * 1e-12
    Rtot = np.random.rand(n_subj, n_ab, n_ag) * np.power(10, np.random.randint(1, 5, size=(n_subj, n_ab, n_ag)))
    Ka = np.random.rand(n_rcp, n_ab) * np.power(10, np.random.randint(5, 8, size=(n_rcp, n_ab)))

    # maserol implementation of binding model
    cube = np.zeros((n_subj, n_rcp, n_ag))
    msRes = inferLbound(cube, Rtot, Ka, L0=L0, KxStar=KxStar, FcIdx=FcIdx)

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
    np.testing.assert_allclose(msRes, vbRes, rtol = 1e-4)

def test_inferLbound_monotonicity():
    # if root finding doesn't converge, the most obvious indicator is commonly a
    # change in the derivative of Lbound as a function of Rtot
    L0 = 1e-10
    KxStar = 1e-12
    ab_types = ("IgG1",)

    Rtot_np = np.random.rand(3000, 1, 1)
    log_steps = 15
    idx_stepsize = int(Rtot_np.shape[0] / log_steps)
    for i in range(log_steps):
        Rtot_np[i * idx_stepsize:(i+1) * idx_stepsize, :, :] *= 10**i
    Rtot = xr.DataArray(Rtot_np, [np.arange(Rtot_np.shape[0]), list(ab_types), np.arange(Rtot_np.shape[2])], ["Sample", "Antibody", "Antigen"],)
    rcp = ["IgG1", "FcgRI"]
    cube = xr.DataArray(np.zeros((Rtot.shape[0], len(rcp), Rtot.shape[2])), (Rtot.Sample.values, rcp, Rtot.Antigen.values),  ("Sample", "Receptor", "Antigen"))
    Ka = assembleKav(cube, ab_types)
    Lbound = inferLbound(cube.values, Rtot.values, Ka.values, L0=L0, KxStar=KxStar, FcIdx=1)

    # Check if the function f is monotonically increasing
    def is_monotonically_increasing(x, y):
        # Sort x and y based on x values
        sorted_indices = np.argsort(x)
        y_sorted = y[sorted_indices]
        y_diff = np.diff(y_sorted)
        return np.all(y_diff >= 0)

    assert is_monotonically_increasing(np.log10(Rtot_np.flatten()), np.log10(Lbound[:, 0, :]))
    assert is_monotonically_increasing(np.log10(Rtot_np.flatten()), np.log10(Lbound[:, 1, :]))


@pytest.mark.parametrize("data", [atyeo(),
                                  SpaceX4D().stack(Sample = ("Subject", "Time"))])
def test_reshape_params(data):
    ab_types = list(HIgGs)
    cube = prepare_data(data)
    abundance, Ka = initializeParams(cube, ab_types=ab_types)
    x = flattenParams(abundance, Ka)
    abundance, Ka = reshapeParams(x, cube, fitKa=True, as_xarray=True, ab_types=ab_types)
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


def generate_random_numbers(n, m):
    # generate n random numbers that sum to m
    random_numbers = np.random.rand(n - 1) * m
    random_numbers.sort()
    random_numbers = np.concatenate(([0], random_numbers, [m]))
    return np.diff(random_numbers)


@pytest.mark.parametrize("n_samp", [8, 32, 256])
@pytest.mark.parametrize("L0", [1e-9, 1e-5])
def test_forward_backward_simple(n_samp, L0):
    # subset of HIgGFs
    ab_types = ["IgG1", "IgG2", "IgG3", "IgG3f"]
    # subset of ['IgG1', 'IgG2', 'IgG3', 'IgG4', 'FcgRI', 'FcgRIIA-131H', 'FcgRIIA-131R',
    #        'FcgRIIB-232I', 'FcgRIIIA-158F', 'FcgRIIIA-158V', 'FcgRIIIB', 'C1q']
    rcp = ['IgG1', 'IgG2', 'IgG3', 'FcgRIIB-232I', 'FcgRIIIA-158F', 'FcgRIIIA-158V', 'FcgRIIIB'] 
    KxStar = 1e-12
    Rtot_np = np.random.rand(n_samp, len(ab_types), 1) * 1e5
    Rtot = xr.DataArray(Rtot_np, [np.arange(Rtot_np.shape[0]), list(ab_types), np.arange(Rtot_np.shape[2])], ["Sample", "Antibody", "Antigen"],)
    cube = xr.DataArray(np.zeros((Rtot.shape[0], len(rcp), Rtot.shape[2])), (Rtot.Sample.values, rcp, Rtot.Antigen.values),  ("Sample", "Receptor", "Antigen"))
    Ka = assembleKav(cube, ab_types)
    cube.values = inferLbound(cube.values, Rtot.values, Ka.values, L0=L0, KxStar=KxStar)
    x_opt, ctx = optimizeLoss(cube, fitKa=False, ab_types=tuple(ab_types), L0=L0,
                    KxStar=KxStar, maxiter=10_000)
    assert ctx["opt"].status > 0
    Rtot_inferred_flat = np.exp(x_opt[:np.prod(Rtot.size)])
    Rtot_flat = Rtot.values.flatten()
    assert np.corrcoef(Rtot_flat, Rtot_inferred_flat)[0][1] > 0.95
