import functools

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from sklearn.metrics import r2_score

from maserol.figures.common import CACHE_DIR, DETECTION_DISPLAY_NAMES, Multiplot
from maserol.forward_backward import add_Ka_noise, forward_backward, perturb_affinity

N_ITER_2D = 3
STEPS_2D = 4
N_ITER_2E = 3
STEPS_2E = 4
DE_YLIM = (0.4, 1.02)
UPDATE_CACHE = {"2b": False, "2c": False, "2d": False, "2e": False, "2f": False}


def makeFigure():
    plot = Multiplot(
        (3, 3),
        fig_size=(7.5, 8),
        subplot_specs=[
            (0, 2, 0, 1),
            (2, 1, 0, 1),
            (0, 1, 1, 1),
            (1, 1, 1, 1),
            (2, 1, 1, 1),
            (0, 3, 2, 1),
        ],
    )
    plot.add_subplot_labels(ax_relative=True)
    axes, fig = plot.axes, plot.fig
    figure_2b(axes[1])
    figure_2c(axes[2])
    figure_2d(axes[3])
    figure_2e(axes[4])
    figure_2f(axes[5])
    fig.tight_layout(pad=0, w_pad=0.2, h_pad=-0.5)
    return fig


def figure_2b(ax):
    if UPDATE_CACHE["2b"]:
        Rtot, Rtot_inferred = forward_backward(0)
        np.savetxt(CACHE_DIR / "figure_2b_Rtot.txt", Rtot.values, "%d")
        np.savetxt(CACHE_DIR / "figure_2b_Rtot_inferred.txt", Rtot_inferred, "%d")
    else:
        Rtot = np.loadtxt(CACHE_DIR / "figure_2b_Rtot.txt")
        Rtot_inferred = np.loadtxt(CACHE_DIR / "figure_2b_Rtot_inferred.txt")
    sns.scatterplot(
        x=Rtot_inferred[:, 0], y=Rtot[:, 0], alpha=0.6, ax=ax
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Actual vs predicted antibody abundance")
    ax.set_xlabel(r"Inferred IgG1")
    ax.set_ylabel(r"Actual IgG1")


def figure_2c(ax):
    noise_std = 0.3
    if UPDATE_CACHE["2c"]:
        Rtot, Rtot_inferred = forward_backward(noise_std)
        np.savetxt(CACHE_DIR / "figure_2c_Rtot.txt", Rtot.values, "%d")
        np.savetxt(CACHE_DIR / "figure_2c_Rtot_inferred.txt", Rtot_inferred, "%d")
    else:
        Rtot = np.loadtxt(CACHE_DIR / "figure_2c_Rtot.txt")
        Rtot_inferred = np.loadtxt(CACHE_DIR / "figure_2c_Rtot_inferred.txt")
    sns.scatterplot(
        x=Rtot_inferred[:, 0], y=Rtot[:, 0], alpha=0.6, ax=ax
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Actual vs predicted (30% noise)")
    ax.set_xlabel(r"Inferred IgG1")
    ax.set_ylabel(r"Actual IgG1")


def figure_2d(ax):
    MAX_NOISE = 0.35
    filename = "figure_2d.csv"
    if UPDATE_CACHE["2d"]:
        noises = np.linspace(0, MAX_NOISE, STEPS_2D)
        dfs = []
        for noise in noises:
            Rtot_pairs = [forward_backward(noise_std=noise) for _ in range(N_ITER_2D)]
            dfs.append(
                pd.DataFrame(
                    {
                        "noise": np.full(N_ITER_2D, noise),
                        "r2": [
                            r2_score(
                                np.log10(Rtot.values.flatten()),
                                np.log10(Rtot_inferred.flatten()),
                            )
                            for Rtot, Rtot_inferred in Rtot_pairs
                        ],
                    }
                )
            )
        df = pd.concat(dfs).reset_index()
        df.to_csv(CACHE_DIR / filename)
    else:
        df = pd.read_csv(CACHE_DIR / filename)

    sns.lineplot(data=df.reset_index(drop=True), x="noise", y="r2", ax=ax)
    ax.set_ylim(DE_YLIM)
    ax.set_xlim((0, MAX_NOISE))
    ax.set_title("Prediction performance vs detection noise")
    xticks = np.linspace(0, MAX_NOISE, STEPS_2D)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.2f}" for x in xticks])
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("$R^2$")
    ax.set_ylim(-0.01, 1.01)


def figure_2e(ax):
    MAX_NOISE = 0.35
    filename = "figure_2e.csv"
    if UPDATE_CACHE["2e"]:
        noises = np.linspace(0, MAX_NOISE, STEPS_2E)
        dfs = []
        for noise in noises:
            Rtot_pairs = [
                forward_backward(
                    Ka_transform_func=functools.partial(add_Ka_noise, noise)
                )
                for _ in range(N_ITER_2E)
            ]
            dfs.append(
                pd.DataFrame(
                    {
                        "noise": np.full(N_ITER_2E, noise),
                        "r2": [
                            r2_score(
                                np.log10(Rtot.values.flatten()),
                                np.log10(Rtot_inferred.flatten()),
                            )
                            for Rtot, Rtot_inferred in Rtot_pairs
                        ],
                    }
                )
            )
        df = pd.concat(dfs).reset_index()
        df.to_csv(CACHE_DIR / filename)
    else:
        df = pd.read_csv(CACHE_DIR / filename)

    sns.lineplot(data=df.reset_index(drop=True), x="noise", y="r2", ax=ax)
    ax.set_ylim(DE_YLIM)
    ax.set_xlim((0, MAX_NOISE))
    ax.set_title("Prediction performance vs $K_{a}$ noise")
    xticks = np.linspace(0, MAX_NOISE, STEPS_2E)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.2f}" for x in xticks])
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("$R^2$")
    ax.set_ylim(-0.01, 1.01)


def figure_2f(ax):
    ligs = ["FcR2A", "FcR2B", "FcR3A", "FcR3B"]
    rcps = ["IgG1", "IgG2", "IgG3", "IgG4"]
    n_runs = 2
    perturbations = [-0.3, 0.3]
    file_path = CACHE_DIR / "2f_perturbations.nc"
    n_cplx = 1000
    if UPDATE_CACHE["2f"]:
        results = xr.DataArray(
            np.nan,
            dims=(
                "Ka_Ligand",
                "Ka_Receptor",
                "Perturbation",
                "Run",
                "Inferred",
                "Complex",
                "Receptor",
            ),
            coords={
                "Ka_Ligand": ligs,
                "Ka_Receptor": rcps,
                "Perturbation": perturbations,
                "Receptor": rcps,
                "Complex": np.arange(n_cplx),
                "Inferred": [True, False],
                "Run": np.arange(n_runs),
            },
        )
        for Ka_lig in ligs:
            for Ka_rcp in rcps:
                for perturbation in perturbations:
                    for run in range(n_runs):
                        Rtot, Rtot_inferred = forward_backward(
                            Ka_transform_func=functools.partial(
                                perturb_affinity, Ka_lig, Ka_rcp, perturbation
                            ),
                            n_cplx=n_cplx,
                        )
                        results.loc[Ka_lig, Ka_rcp, perturbation, run, True] = (
                            Rtot_inferred
                        )
                        results.loc[Ka_lig, Ka_rcp, perturbation, run, False] = Rtot
                        results.to_netcdf(file_path)
    results = xr.open_dataarray(CACHE_DIR / "2f_perturbations.nc")
    performance = pd.DataFrame(
        columns=rcps,
        index=pd.MultiIndex.from_tuples([], names=["Ka_Ligand", "Ka_Receptor"]),
    )
    for Ka_lig in ligs:
        for Ka_rcp in rcps:
            for rcp in rcps:
                r2 = []
                for perturbation in perturbations:
                    for run in range(n_runs):
                        r2.append(
                            r2_score(
                                np.log10(
                                    results.loc[
                                        Ka_lig, Ka_rcp, perturbation, run, False, :, rcp
                                    ]
                                ),
                                np.log10(
                                    results.loc[
                                        Ka_lig, Ka_rcp, perturbation, run, True, :, rcp
                                    ]
                                ),
                            )
                        )
                performance.loc[(Ka_lig, Ka_rcp), rcp] = np.mean(np.array(r2))
    performance = performance.astype(float)
    # join multiindex
    performance.index = performance.index.map(
        lambda x: f"{x[1]}-{DETECTION_DISPLAY_NAMES[x[0]]}"
    )
    performance = performance.sort_index()
    sns.heatmap(1 - performance.values.T, ax=ax, vmin=0)
    # remove colorbar
    cbar = ax.collections[0].colorbar
    cbar.remove()
    ax.set_ylabel("Antibody")
    ax.set_xlabel("$K_a$")
    ax.set_yticklabels(rcps, rotation=0)
    ax.set_xticklabels(performance.index, rotation=90)
