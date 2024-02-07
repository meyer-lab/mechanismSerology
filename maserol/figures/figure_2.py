from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

from maserol.figures.common import getSetup, add_subplot_labels, CACHE_DIR
from maserol.forward_backward import forward_backward

N_ITER_2D = 3
STEPS_2D = 4
N_ITER_2E = 3
STEPS_2E = 4
DE_YLIM = (0.4, 1.02)
UPDATE_CACHE = {"2b": False, "2c": False, "2d": False, "2e": False}


def makeFigure():
    axes, fig = getSetup((3 * 3, (2.5 * 2)), (2, 3))
    figure_2b(axes[1])
    figure_2c(axes[2])
    figure_2d(axes[3])
    figure_2e(axes[4])
    add_subplot_labels(axes[0:5])
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
        x=np.log10(Rtot_inferred[:, 0]), y=np.log10(Rtot[:, 0]), alpha=0.6, ax=ax
    )
    ax.set_title("Actual vs predicted antibody abundance")
    ax.set_xlabel("log10 Inferred IgG1")
    ax.set_ylabel("log10 Actual IgG1")


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
        x=np.log10(Rtot_inferred[:, 0]), y=np.log10(Rtot[:, 0]), alpha=0.6, ax=ax
    )
    ax.set_title("Actual vs predicted antibody abundance (30% noise)")
    ax.set_xlabel("log10 Inferred IgG1")
    ax.set_ylabel("log10 Actual IgG1")


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
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("$r^2$")


def figure_2e(ax):
    MAX_NOISE = 0.35
    filename = "figure_2e.csv"
    if UPDATE_CACHE["2e"]:
        noises = np.linspace(0, MAX_NOISE, STEPS_2E)
        dfs = []
        for noise in noises:
            Rtot_pairs = [
                forward_backward(Ka_noise_std=noise) for _ in range(N_ITER_2E)
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
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("$r^2$")
