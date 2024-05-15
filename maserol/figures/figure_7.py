import ast
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from maserol.datasets import Zohar
from maserol.figures.common import (CACHE_DIR, DETECTION_DISPLAY_NAMES,
                                    Multiplot)
from maserol.impute import assemble_residual_mask, impute_missing_ms
from maserol.util import assemble_options

UPDATE_CACHE = {"metrics": False, "scatter": [False, False]}
RUNS = 3

METRIC_LABEL_MAPPINGS = {
    "r": "$r$",
    "r2": "$R^2$",
}


TITLE_FONT_SIZE = 12.5


def makeFigure():
    plot = Multiplot(
        (4, 8),
        fig_size=(7.5, 7.5),
        subplot_specs=[
            (0, 2, 0, 3),
            (2, 2, 0, 3),
            (0, 2, 3, 3),
            (2, 2, 3, 3),
            (0, 1, 6, 2),
            (1, 1, 6, 2),
            (2, 1, 6, 2),
            (3, 1, 6, 2),
        ],
    )

    if UPDATE_CACHE["metrics"]:
        update_cache(2)
        update_cache(3)

    df_2 = pd.read_csv(CACHE_DIR / "assay_optimize_2_lig.csv")
    df_3 = pd.read_csv(CACHE_DIR / "assay_optimize_3_lig.csv")

    prepare_metrics_df(df_2)
    prepare_metrics_df(df_3)

    plot_combinations(df_2, "r", plot.axes[0], ylim=(0, 1))
    plot_combinations(df_2, "r2", plot.axes[2], ylim=(-1, 1))

    plot_combinations(df_3, "r", plot.axes[1], ylim=(0, 1))
    plot_combinations(df_3, "r2", plot.axes[3], ylim=(-1, 1))

    plot.axes[1].set_yticklabels([])
    plot.axes[1].set_ylabel(None)
    plot.axes[3].set_ylabel(None)
    plot.axes[3].set_yticklabels([])

    scatter(["FcR2B", "FcR3B"], plot.axes[4:6], UPDATE_CACHE["scatter"][0])
    scatter(["FcR3A", "FcR3B"], plot.axes[6:8], UPDATE_CACHE["scatter"][1])
    for ax in plot.axes[5:]:
        ax.set_ylabel(None)

    plot.add_subplot_labels()

    return plot.fig


def plot_combinations(df, metric, ax, ylim=None):
    combs = df["comb"].unique()
    rligs = set()
    for comb in combs:
        for lig in comb:
            rligs.add(lig)
    rligs = list(rligs)
    df_sub = df[df["metric"] == metric]
    combs = df_sub["comb"].unique()
    lig_idxs = df_sub["lig"].unique()

    n_comb = len(combs)
    n_lig = len(lig_idxs)

    width = 1 / (n_lig + 2)

    palette = sns.color_palette(n_colors=len(rligs))
    colors = {lig: palette[i] for i, lig in enumerate(rligs)}

    for i, comb in enumerate(combs):
        for lig_idx, lig in enumerate(comb):
            data = df_sub[(df_sub["comb"] == comb) & (df_sub["lig"] == lig_idx)]["val"]
            position = i - (n_lig - 1) / 2 * width + lig_idx * width
            mean = np.mean(data)
            std = np.std(data)
            ax.bar(position, mean, width, color=colors[lig], edgecolor="black")
            ax.errorbar(position, mean, yerr=std, fmt="none", color="black", capsize=3)

    offsets = [width / 2 * -(n_lig - 1) + i * width for i in range(n_lig)]
    central_positions = [i + offset for i in range(n_comb) for offset in offsets]

    ax.set_xticks(central_positions)
    ax.set_xticklabels(
        [DETECTION_DISPLAY_NAMES[lig] for comb in combs for lig in comb], rotation=45
    )
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.xaxis.grid(False)

    return plt


def prepare_metrics_df(df):
    combs = []
    for i in df.index:
        combs.append(ast.literal_eval(df.loc[i, "comb"]))
    df["comb"] = combs


def update_cache(n_crligs):
    detection_signal = Zohar().get_detection_signal()

    rligs = [l for l in detection_signal.Ligand.values if "Fc" in l]

    opts = assemble_options(detection_signal)

    df = pd.DataFrame(columns=["comb", "lig", "metric", "val"])

    for crligs in itertools.combinations(rligs, n_crligs):
        for _ in range(RUNS):
            # select subset for speed
            residual_mask = assemble_residual_mask(detection_signal, {tuple(crligs): 1})
            actual = np.log10(detection_signal.values + 1)
            Lbound = np.log10(
                impute_missing_ms(detection_signal, residual_mask, opts) + 1
            )
            for i, lig in enumerate(crligs):
                idx = list(detection_signal.Ligand.values).index(lig)
                df.loc[len(df)] = [
                    crligs,
                    i,
                    "r",
                    pearsonr(actual[:, idx], Lbound[:, idx]).statistic,
                ]
                df.loc[len(df)] = [
                    crligs,
                    i,
                    "r2",
                    r2_score(actual[:, idx], Lbound[:, idx]),
                ]
            df.to_csv(CACHE_DIR / f"assay_optimize_{n_crligs}_lig_backup.csv")

    df.to_csv(CACHE_DIR / f"assay_optimize_{n_crligs}_lig.csv")


def scatter(ligs, axes, update_cache):
    detection_signal = Zohar().get_detection_signal()
    opts = assemble_options(detection_signal)

    residual_mask = assemble_residual_mask(detection_signal, {tuple(ligs): 1})
    file_path = CACHE_DIR / f"fig_7_{'_'.join(ligs)}.csv"
    if update_cache:
        Lbound = impute_missing_ms(detection_signal, residual_mask, opts)
        np.savetxt(file_path, Lbound, fmt="%f")
    else:
        Lbound = np.loadtxt(file_path, dtype=float)
    for i, lig in enumerate(ligs):
        idx = list(detection_signal.Ligand.values).index(lig)
        subset = np.random.choice(Lbound.shape[0], 1300, replace=False)
        inferred = np.log10(Lbound[subset, idx] + 1)
        measured = np.log10(detection_signal[subset, idx] + 1)
        sns.scatterplot(x=inferred, y=measured, ax=axes[i], alpha=0.4, s=5)
        axes[i].set_title(f"{DETECTION_DISPLAY_NAMES[lig]}")
        axes[i].set_ylabel("$\mathrm{log_{10}}$ Measured")
        axes[i].set_xlabel("$\mathrm{log_{10}}$ Inferred")
        # annotate with r2 and r
        axes[i].text(
            0.65,
            0.15,
            f"$R^2$={r2_score(measured, inferred):.2f}",
            transform=axes[i].transAxes,
        )
        axes[i].text(
            0.65,
            0.05,
            f"$r$={pearsonr(measured, inferred)[0]:.2f}",
            transform=axes[i].transAxes,
        )
        # set ticks
        axes[i].set_xticks(np.arange(0, 7))
        axes[i].set_yticks(np.arange(0, 7))
        axes[i].set_xlim(0, 7)
        axes[i].set_ylim(0, 7)
        axes[i].plot([0, 7], [0, 7], color="black", linestyle="--", alpha=0.4)
