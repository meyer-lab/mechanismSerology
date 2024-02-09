import numpy as np
import xarray as xr

from maserol.core import infer_Lbound_mv, optimize_loss, reshape_params
from maserol.util import assemble_Ka, assemble_options


def forward_backward(noise_std=0, Ka_noise_std=0, tol=1e-5):
    n_cplx = int(1000)

    rcps = ["IgG1", "IgG2", "IgG3", "IgG4"]
    n_rcp = len(rcps)

    lig = [
        "IgG1",
        "IgG2",
        "IgG3",
        "IgG4",
        "FcR2A",
        "FcR2B",
        "FcR3A",
        "FcR3B",
    ]
    n_lig = len(lig)

    Rtot = xr.DataArray(
        10 ** np.random.normal(loc=2, scale=0.4, size=(n_cplx, n_rcp)),
        [np.arange(n_cplx), list(rcps)],
        ["Complex", "Receptor"],
    )
    data = xr.DataArray(
        np.zeros((n_cplx, n_lig)), (Rtot.Complex.values, lig), ("Complex", "Ligand")
    )

    backward_opts = assemble_options(data, rcps)
    backward_opts["tol"] = tol

    forward_Ka = np.ones((n_lig, n_rcp), dtype=float)
    forward_Ka[4:] = assemble_Ka(data.Ligand.values[4:], rcps).values
    forward_Ka[np.arange(4), np.arange(4)] = 1e7
    forward_Ka *= np.maximum(
        1 + np.random.normal(scale=Ka_noise_std, size=forward_Ka.shape), 0
    )
    forward_L0 = np.concatenate((np.full(4, 2e-9), backward_opts["L0"]))
    forward_KxStar = np.concatenate((np.full(4, 1e-12), backward_opts["KxStar"]))
    forward_f = np.concatenate((np.full(4, 2), backward_opts["f"]))
    data.values = infer_Lbound_mv(
        Rtot.values,
        forward_Ka,
        forward_L0,
        forward_KxStar,
        forward_f,
    )

    # lognormal
    # data.values *= 10 ** np.random.normal(scale=noise_std, size=data.shape)

    # normal
    data.values *= np.maximum(1 + np.random.normal(scale=noise_std, size=data.shape), 0)

    x_opt, ctx = optimize_loss(data, **backward_opts)
    assert ctx["opt"].status > 0
    params = reshape_params(x_opt, data, backward_opts["logistic_ligands"], rcps=rcps)

    return Rtot, params["Rtot"]
