import pytest
import numpy as np
import xarray as xr
from valentbind.model import polyc

from maserol.core import (
    initialize_params,
    infer_Lbound_mv,
    assemble_Ka,
    n_logistic_ligands,
    reshape_params,
    optimize_loss,
)
from maserol.datasets import Zohar
from maserol.util import assemble_options, HIgGs


@pytest.mark.parametrize("n_rcp", [1, 2, 3])
def test_initialize(n_rcp):
    detection_signal = Zohar().get_detection_signal()
    opts = assemble_options(detection_signal)
    n_cplx, n_lig = detection_signal.shape
    ps = initialize_params(detection_signal, opts["logistic_ligands"], opts["rcps"])
    assert ps["logistic_params"].shape == (
        4,
        n_logistic_ligands(opts["logistic_ligands"]),
    )


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


@pytest.mark.parametrize("n_cplx", [50])
@pytest.mark.parametrize("L0", [1e-9, 1e-5])
@pytest.mark.parametrize("rcp_high", [1e3, 1e7])
def test_forward_backward(n_cplx, L0, rcp_high):
    rcps = np.array(["IgG1", "IgG2", "IgG3", "IgG3f"])
    ligs = np.array(
        [
            "IgG1",
            "IgG2",
            "IgG3",
            "FcR2A",
            "FcR2B",
            "FcR3A",
            "FcR3B",
        ]
    )
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
    Ka = np.ones((len(ligs), len(rcps)))
    Ka[3:] = assemble_Ka(ligs[3:], rcps).values
    Ka[[0, 1, 2, 2], [0, 1, 2, 3]] = 10**7

    backward_opts = assemble_options(data, rcps)

    forward_L0 = np.concatenate((np.full(3, 2e-9), backward_opts["L0"]))
    forward_KxStar = np.concatenate((np.full(3, 1e-12), backward_opts["KxStar"]))
    forward_f = np.concatenate((np.full(3, 2), backward_opts["f"]))
    data.values = infer_Lbound_mv(
        Rtot.values, Ka, forward_L0, forward_KxStar, forward_f
    )

    x_opt, ctx = optimize_loss(
        data,
        backward_opts["L0"],
        backward_opts["KxStar"],
        backward_opts["f"],
        rcps,
        backward_opts["logistic_ligands"],
        Ka=Ka[3:],
    )

    assert ctx["opt"].status > 0

    params = reshape_params(x_opt, data, backward_opts["logistic_ligands"], rcps)

    assert np.corrcoef(Rtot.values.flatten(), params["Rtot"].flatten())[0][1] > 0.95
