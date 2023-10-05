import pytest
import numpy as np
import xarray as xr
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data as zohar
from tensordata.kaplonek import SpaceX4D
from valentbind.model import polyc

from maserol.core import (
    initialize_params,
    infer_Lbound_mv,
    assemble_Ka,
    n_logistic_ligands,
    reshape_params,
    optimize_loss,
)
from maserol.preprocess import assemble_options, HIgGs, prepare_data


@pytest.mark.parametrize("n_rcp", [1, 2, 3])
def test_initialize(n_rcp):
    data = prepare_data(zohar())
    opts = assemble_options(data, HIgGs[:n_rcp])
    n_cplx, n_lig = data.shape
    ps = initialize_params(data, opts["logistic_ligands"], opts["rcps"], fitKa=True)
    assert ps["Rtot"].shape == (n_cplx, n_rcp)
    assert ps["logistic_params"].shape == (
        4,
        n_logistic_ligands(opts["logistic_ligands"]),
    )
    assert ps["Ka"].shape == (n_lig, n_rcp)


def test_inferLbound_matches_valentbind():
    """Test that this implementation of infer_Lbound matches the implementation
    in the valentbind package."""
    n_cplx, n_lig, n_rcp = 50, 4, 3
    L0 = np.random.uniform(1e-7, 1e-11, n_lig)
    KxStar = np.random.uniform(1e-9, 1e-15, n_lig)
    f = np.random.choice(np.arange(1, 20), n_lig)
    Rtot = np.random.rand(n_cplx, n_rcp) * np.power(
        10, np.random.randint(1, 5, size=(n_cplx, n_rcp))
    )
    Ka = np.random.rand(n_lig, n_rcp) * np.power(
        10, np.random.randint(5, 8, size=(n_lig, n_rcp))
    )

    # maserol implementation of binding model
    msRes = infer_Lbound_mv(Rtot, Ka, L0, KxStar, f)

    # valentbind implementation of binding model
    vbRes = np.zeros((n_cplx, n_lig))
    for i_cplx in range(n_cplx):
        for i_lig in range(n_lig):
            vbRes[i_cplx, i_lig] = polyc(
                L0[i_lig],
                KxStar[i_lig],
                Rtot[i_cplx, :],
                np.array([[f[i_lig]]]),  # f
                np.array([1]),  # Ctheta
                Ka[[i_lig], :],
            )[0]

    # compare
    np.testing.assert_allclose(msRes, vbRes, rtol=1e-4)


@pytest.mark.parametrize("n_cplx", [10, 400])
@pytest.mark.parametrize("L0", [1e-9, 1e-5])
@pytest.mark.parametrize("rcp_high", [1e3, 1e7])
def test_forward_backward(n_cplx, L0, rcp_high):
    rcps = ["IgG1", "IgG2", "IgG3", "IgG3f"]
    ligs = [
        "IgG1",
        "IgG2",
        "IgG3",
        "FcgRIIB-232I",
        "FcgRIIIA-158F",
        "FcgRIIIA-158V",
        "FcgRIIIB",
    ]
    f = np.array([2, 2, 2, 4, 4, 4, 4])
    L0 = np.full(len(ligs), 1e-9)
    KxStar = np.full(len(ligs), 1e-12)
    Rtot = xr.DataArray(
        np.random.uniform(high=rcp_high, size=(n_cplx, len(rcps))),
        [np.arange(n_cplx), list(rcps)],
        ["Complex", "Receptor"],
    )
    data = xr.DataArray(
        np.zeros((n_cplx, len(ligs))),
        (Rtot.Complex.values, ligs),
        ("Complex", "Ligand"),
    )
    Ka = assemble_Ka(data, rcps)
    data.values = infer_Lbound_mv(Rtot.values, Ka.values, L0, KxStar, f)
    opts = assemble_options(data, rcps)
    x_opt, ctx = optimize_loss(data, **opts)
    assert ctx["opt"].status > 0
    params = reshape_params(x_opt, data, opts["logistic_ligands"])
    assert np.corrcoef(Rtot.values.flatten(), params["Rtot"].flatten())[0][1] > 0.95
