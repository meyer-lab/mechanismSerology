import ast
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from maserol.datasets import Zohar
from maserol.figures.common import CACHE_DIR, Multiplot, DETECTION_DISPLAY_NAMES
from maserol.impute import assemble_residual_mask, impute_missing_ms
from maserol.util import assemble_options

UPDATE_CACHE = False
RUNS = 3

METRIC_LABEL_MAPPINGS = {
    "r": "$r$",
    "r2": "$R^2$",
}


TITLE_FONT_SIZE = 13


def makeFigure():
    plot = Multiplot((2, 2), (4.5, 3))

    if UPDATE_CACHE:
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
    plot.axes[0].set_title("2 Missing Detections", fontsize=TITLE_FONT_SIZE)
    plot.axes[1].set_title("3 Missing Detections", fontsize=TITLE_FONT_SIZE)
    plot.axes[1].set_ylabel(None)
    plot.axes[3].set_ylabel(None)

    plot.axes[0].set_ylabel(
        f"Imputation Accuracy ({METRIC_LABEL_MAPPINGS['r']})", fontsize=TITLE_FONT_SIZE
    )
    plot.axes[2].set_ylabel(
        f"Imputation Accuracy ({METRIC_LABEL_MAPPINGS['r2']})", fontsize=TITLE_FONT_SIZE
    )

    plot.add_subplot_labels()
    plot.fig.tight_layout()

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
            actual = np.log10(detection_signal.values)
            Lbound = np.log10(impute_missing_ms(detection_signal, residual_mask, opts))
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
