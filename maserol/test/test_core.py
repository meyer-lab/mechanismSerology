import pytest
import numpy as np
import xarray as xr
from tensordata.atyeo import data as atyeo
from tensordata.kaplonek import SpaceX4D
from valentbind.model import polyc

from ..core import initializeParams, inferLbound, assembleKav, flattenParams, reshapeParams, optimizeLoss
from ..preprocess import HIgGs, prepare_data


@pytest.mark.parametrize("n_ab", [1, 2, 3])
def test_initialize(n_ab):
    """ Test initializeParams() work correctly """
    cube = prepare_data(atyeo())
    n_samp, n_rcp, n_ag = cube.shape
    ab_types = HIgGs[:n_ab]
    ps = initializeParams(cube, ab_types=ab_types)  # should return abund, Ka
    assert len(ps) == 2
    assert ps[0].shape == (n_samp, n_ab, n_ag)
    assert ps[1].shape == (n_rcp, n_ab)


def test_inferLbound_matches_valentbind():
    """ Test that our model here provides the same outcome as expected """
    n_subj, n_rcp, n_ag, n_ab = 6, 50, 4, 3
    L0 = np.random.uniform(1e-7, 1e-11, n_rcp)
    KxStar = np.random.uniform(1e-9, 1e-15, n_rcp)
    f = np.random.choice(np.arange(1, 20), n_rcp)
    Rtot = np.random.rand(n_subj, n_ab, n_ag) * np.power(10, np.random.randint(1, 5, size=(n_subj, n_ab, n_ag)))
    Ka = np.random.rand(n_rcp, n_ab) * np.power(10, np.random.randint(5, 8, size=(n_rcp, n_ab)))

    # maserol implementation of binding model
    cube = np.zeros((n_subj, n_rcp, n_ag))
    msRes = inferLbound(cube, Rtot, Ka, L0, KxStar, f)

    # valentbind implementation of binding model
    vbRes = np.zeros((n_subj, n_rcp, n_ag))
    for i_subj in range(n_subj):
        for i_rcp in range(n_rcp):
            for i_ag in range(n_ag):
                vbRes[i_subj, i_rcp, i_ag] = polyc(
                    L0[i_rcp], KxStar[i_rcp], Rtot[i_subj, : , i_ag],
                    np.array([[f[i_rcp]]]), # f
                    np.array([1]), # Ctheta
                    Ka[[i_rcp], :])[0]
    
    # compare
    np.testing.assert_allclose(msRes, vbRes, rtol = 1e-4)

def test_inferLbound_monotonicity():
    # if root finding doesn't converge, sometimes a good indicator is that
    # inferLbound does not monotonically increase
    ab_types = ("IgG1",)

    Rtot_np = np.random.rand(3000, 1, 1)
    log_steps = 15
    idx_stepsize = int(Rtot_np.shape[0] / log_steps)
    for i in range(log_steps):
        Rtot_np[i * idx_stepsize:(i+1) * idx_stepsize, :, :] *= 10**i
    Rtot = xr.DataArray(Rtot_np, [np.arange(Rtot_np.shape[0]), list(ab_types), np.arange(Rtot_np.shape[2])], ["Sample", "Antibody", "Antigen"],)
    rcp = np.array(["IgG1", "FcgRI"])
    cube = xr.DataArray(np.zeros((Rtot.shape[0], rcp.size, Rtot.shape[2])), (Rtot.Sample.values, rcp, Rtot.Antigen.values),  ("Sample", "Receptor", "Antigen"))
    Ka = assembleKav(cube, ab_types)
    Lbound = inferLbound(cube.values, Rtot.values, Ka.values, np.full(rcp.size, 1e-9), np.full(rcp.size, 1e-12), np.full(rcp.size, 4))

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

@pytest.mark.parametrize("n_samp", [10, 100, 1000])
@pytest.mark.parametrize("L0", [1e-9, 1e-5])
@pytest.mark.parametrize("rcp_high", [1e2, 1e5, 1e8])
def test_forward_backward_simple(n_samp, L0, rcp_high):
    # subset of HIgGFs
    ab_types = ["IgG1", "IgG2", "IgG3", "IgG3f"]
    # subset of ['IgG1', 'IgG2', 'IgG3', 'IgG4', 'FcgRI', 'FcgRIIA-131H', 'FcgRIIA-131R',
    #        'FcgRIIB-232I', 'FcgRIIIA-158F', 'FcgRIIIA-158V', 'FcgRIIIB', 'C1q']
    rcp = ['IgG1', 'IgG2', 'IgG3', 'FcgRIIB-232I', 'FcgRIIIA-158F', 'FcgRIIIA-158V', 'FcgRIIIB'] 
    f = np.array([2, 2, 2, 4, 4, 4, 4])
    L0 = np.full(len(rcp), 1e-9)
    KxStar = np.full(len(rcp), 1e-12)
    Rtot_np = np.random.uniform(high=rcp_high, size=(n_samp, len(ab_types), 1))
    Rtot = xr.DataArray(Rtot_np, [np.arange(Rtot_np.shape[0]), list(ab_types), np.arange(Rtot_np.shape[2])], ["Sample", "Antibody", "Antigen"],)
    cube = xr.DataArray(np.zeros((Rtot.shape[0], len(rcp), Rtot.shape[2])), (Rtot.Sample.values, rcp, Rtot.Antigen.values),  ("Sample", "Receptor", "Antigen"))
    Ka = assembleKav(cube, ab_types)
    cube.values = inferLbound(cube.values, Rtot.values, Ka.values, L0, KxStar, f)
    x_opt, ctx = optimizeLoss(cube, L0, KxStar, f, tuple(ab_types))
    assert ctx["opt"].status > 0
    Rtot_inferred_flat = np.exp(x_opt[:np.prod(Rtot.size)])
    Rtot_flat = Rtot.values.flatten()
    assert np.corrcoef(Rtot_flat, Rtot_inferred_flat)[0][1] > 0.95


def test_fit_Ka():
    ab_types = HIgGs
    cube = prepare_data(atyeo())
    n_rcp = cube.sizes["Receptor"]
    Ka = assembleKav(cube, ab_types)
    L0 = np.full(n_rcp, 1e-9)
    KxStar = np.full(n_rcp, 1e-12)
    f = np.full(n_rcp, 4) 
    # fitKa contains the idx for the affinities in Ka which should be fit
    idx = [(0, 0), (1, 1)] # fit the affinities at (0, 0) and (1, 1)
    fitKa = tuple(map(np.array, zip(*idx)))
    x_opt, _ = optimizeLoss(cube, L0, KxStar, f, ab_types, fitKa=fitKa)
    Rtot, Ka_fitted = reshapeParams(x_opt, cube, fitKa=fitKa, ab_types=ab_types)
    Ka_after = Ka.copy()
    Ka_after.values[fitKa] = Ka_fitted
    assert Ka_after[0, 0] != Ka[0, 0]
    assert Ka_after[1, 1] != Ka[1, 1]
    mask = np.ones(Ka.shape, dtype=int)
    mask[fitKa] = 0
    fitKa_not = np.where(mask)
    assert (Ka_after.values[fitKa_not] == Ka.values[fitKa_not]).all()
