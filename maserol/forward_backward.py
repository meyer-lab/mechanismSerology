from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr

from maserol.core import infer_Lbound_mv, optimize_loss, reshape_params
from maserol.util import assemble_Ka, assemble_options


def generate_rtot_distribution(n_samples, n_receptors=4):
    """
    Generate the Rtot distribution used in forward_backward.

    Args:
    n_samples (int): Number of samples to generate.
    n_receptors (int): Number of receptors.

    Returns:
    numpy.ndarray: Array of shape (n_samples, n_receptors) containing the
    generated distribution.
    """
    return 10 ** np.random.normal(loc=2, scale=0.4, size=(n_samples, n_receptors))


def forward_backward(
    noise_std=0,
    Ka_transform_func=lambda x: x,
    tol=1e-5,
    n_cplx=1000,
    KxStar_perturb=None,
) -> tuple[np.ndarray, pd.DataFrame]:
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
        generate_rtot_distribution(n_cplx, n_rcp),
        [np.arange(n_cplx), list(rcps)],
        ["Complex", "Receptor"],
    )
    data = xr.DataArray(
        np.zeros((n_cplx, n_lig)), (Rtot.Complex.values, lig), ("Complex", "Ligand")
    )

    backward_opts = assemble_options(data, rcps)
    backward_opts["tol"] = tol

    forward_Ka = xr.DataArray(
        np.ones((n_lig, n_rcp), dtype=float), [lig, rcps], ["Ligand", "Receptor"]
    )
    forward_Ka.values[4:] = assemble_Ka(data.Ligand.values[4:], rcps).values
    forward_Ka.values[np.arange(4), np.arange(4)] = 1e7

    forward_Ka = Ka_transform_func(forward_Ka).values

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

    if KxStar_perturb is not None:
        n_segs = len(KxStar_perturb)
        seg_size = n_cplx // n_segs
        Rtot_inferred_list = []

        for i in range(KxStar_perturb.size):
            backward_opts_seg = deepcopy(backward_opts)
            backward_opts_seg["KxStar"] *= KxStar_perturb[i]

            if i == n_segs - 1:
                data_seg = data[i * seg_size :]
            else:
                data_seg = data[i * seg_size : (i + 1) * seg_size]

            x_opt, ctx = optimize_loss(data_seg, **backward_opts_seg)
            assert ctx["opt"].status > 0
            params = reshape_params(
                x_opt, data_seg, backward_opts["logistic_ligands"], rcps=rcps
            )
            Rtot_inferred_seg = params["Rtot"]
            Rtot_inferred_df = pd.DataFrame(Rtot_inferred_seg, columns=rcps)
            Rtot_inferred_df["Perturbation"] = KxStar_perturb[i]
            Rtot_inferred_list.append(Rtot_inferred_df)

        Rtot_inferred = pd.concat(Rtot_inferred_list, ignore_index=True)
    else:
        x_opt, ctx = optimize_loss(data, **backward_opts)
        assert ctx["opt"].status > 0
        params = reshape_params(
            x_opt, data, backward_opts["logistic_ligands"], rcps=rcps
        )
        Rtot_inferred = params["Rtot"]

    return Rtot, Rtot_inferred


def add_Ka_noise(noise_std: float, Ka: xr.DataArray):
    return Ka * np.maximum(1 + np.random.normal(scale=noise_std, size=Ka.shape), 0)


def perturb_affinity(lig: str, rcp: str, perturbation: float, Ka: xr.DataArray):
    Ka_perturbed = Ka.copy()
    Ka_perturbed.loc[dict(Ligand=lig, Receptor=rcp)] = Ka.sel(
        Ligand=lig, Receptor=rcp
    ) * (1 + perturbation)
    return Ka_perturbed
