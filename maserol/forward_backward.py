import numpy as np
import xarray as xr

from maserol.core import infer_Lbound_mv, optimize_loss, reshape_params
from maserol.preprocess import assemble_Ka, assemble_options


def forward_backward(noise_std=0, Ka_noise_std=0, tol=1e-5):
    n_cplx = int(1000)

    rcps = ["IgG1", "IgG2", "IgG3", "IgG4"]
    n_rcp = len(rcps)

    lig = [
        "IgG1",
        "IgG2",
        "IgG3",
        "IgG4",
        "FcgRIIA-131R",
        "FcgRIIB-232I",
        "FcgRIIIA",
        "FcgRIIIB",
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
    Ka = assemble_Ka(data, rcps)
    opts = assemble_options(data, rcps)
    opts["tol"] = tol

    forward_Ka = Ka.values * np.maximum(
        1 + np.random.normal(scale=Ka_noise_std, size=Ka.shape), 0
    )
    data.values = infer_Lbound_mv(
        Rtot.values, forward_Ka, opts["L0"], opts["KxStar"], opts["f"]
    )

    # lognormal
    # data.values *= 10 ** np.random.normal(scale=noise_std, size=data.shape)

    # normal
    data.values *= np.maximum(1 + np.random.normal(scale=noise_std, size=data.shape), 0)

    x_opt, ctx = optimize_loss(data, **opts)
    assert ctx["opt"].status > 0
    params = reshape_params(x_opt, data, opts["logistic_ligands"], rcps=rcps)

    return Rtot, params["Rtot"]
