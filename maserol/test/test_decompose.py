import numpy as np
import xarray as xr

from maserol.decompose import factorAbundance, reconstructAbundance


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
        coords=(np.arange(n_sample), np.arange(n_ab), np.arange(n_ag)),
    )
    got_sample_facs, got_ag_facs = factorAbundance(
        abundance_xr, n_comps, as_xarray=False
    )

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
    assert normalized_error(got_sample_facs, sample_facs) < 0.6 * normalized_error(
        baseline, sample_facs
    )

    got_abundance = reconstructAbundance(got_sample_facs, got_ag_facs)
    baseline = np.random.rand(*abundance.shape) * np.mean(abundance)
    # assert that the reconstruction error is small
    assert normalized_error(got_abundance, abundance) < 0.01 * normalized_error(
        baseline, abundance
    )
