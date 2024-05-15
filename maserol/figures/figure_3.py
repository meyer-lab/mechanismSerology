import functools

import numpy as np
import pandas as pd
import seaborn as sns

from maserol.datasets import Zohar
from maserol.figures.common import CACHE_DIR, DETECTION_DISPLAY_NAMES, Multiplot
from maserol.impute import (
    assemble_residual_mask,
    imputation_scatterplot,
    impute_missing_ms,
    impute_missing_pca,
    run_repeated_imputation,
)

UPDATE_CACHE = {"3b": False, "3c": False, "3e": False}
LIGS_3C = ["IgG1", "IgG3", "FcR2A", "FcR2B", "FcR3A", "FcR3B"]
RUNS_PER_LIG_3C = 3


def makeFigure():
    plot = Multiplot(
        (9, 3),
        fig_size=(7.5, 7.7),
        subplot_specs=[
            (0, 6, 0, 1),
            (6, 3, 0, 1),
            (0, 3, 1, 1),
            (3, 3, 1, 1),
            (6, 3, 1, 1),
            (0, 3, 2, 1),
            (3, 3, 2, 1),
            (6, 3, 2, 1),
        ],
    )
    figure_3b(plot.axes[2])
    figure_3cd(plot.axes[3], plot.axes[4])
    figure_3e(plot.axes[6], plot.axes[7])
    plot.fig.tight_layout(pad=0.01, w_pad=0, h_pad=0.2)
    plot.add_subplot_labels(ax_relative=True)
    return plot.fig


def figure_3b(ax):
    inferred_filename = "fig_3b_imputation.txt"
    residual_mask_filename = "fig_3b_res_mask.txt"
    data = Zohar().get_detection_signal()
    if UPDATE_CACHE["3b"]:
        missingness = {"FcR3B": 0.1}
        residual_mask = assemble_residual_mask(data, missingness)
        Lbound = impute_missing_ms(data, residual_mask)
        np.savetxt(CACHE_DIR / inferred_filename, Lbound, fmt="%d")
        np.savetxt(CACHE_DIR / residual_mask_filename, residual_mask, fmt="%d")
    else:
        Lbound = np.loadtxt(CACHE_DIR / inferred_filename, dtype=float)
        residual_mask = np.loadtxt(CACHE_DIR / residual_mask_filename, dtype=bool)
    imputation_scatterplot(data, Lbound, residual_mask, ax)
    lim = (0.7, 6.9)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks(np.arange(1, 7))
    ax.set_yticks(np.arange(1, 7))
    ax.set_xlabel(r"$\mathrm{log_{10}}$ Inferred " + DETECTION_DISPLAY_NAMES["FcR3B"])
    ax.set_ylabel(r"$\mathrm{log_{10}}$ Measured " + DETECTION_DISPLAY_NAMES["FcR3B"])
    ax.set_title("10% Missing")
    ax.plot(lim, lim, linestyle="--", color="gray", alpha=0.75)


def figure_3cd(ax_c, ax_d):
    filename = "fig_3c_metrics.csv"
    if UPDATE_CACHE["3c"]:
        N_COMP = 1
        data = Zohar().get_detection_signal()
        imputers = [
            impute_missing_ms,
            functools.partial(impute_missing_pca, ncomp=N_COMP),
        ]
        df = pd.concat(
            run_repeated_imputation(
                data, imputer, missingness=0.1, ligs=LIGS_3C, runs=RUNS_PER_LIG_3C
            )
            for imputer in imputers
        )
        df.to_csv(CACHE_DIR / filename)
    else:
        df = pd.read_csv(CACHE_DIR / filename)

    df = df.replace({"impute_missing_pca": "PCA", "impute_missing_ms": "Binding model"})

    sns.barplot(data=df, x="Ligand", y="r", hue="Method", ax=ax_c)
    ax_c.set_xticklabels(
        [DETECTION_DISPLAY_NAMES[label._text] for label in ax_c.get_xticklabels()],
        rotation=45,
    )
    ax_c.set_xlabel("Detection")
    ax_c.set_ylabel("Imputation performance ($r$)")
    ax_c.set_title("10% Missing")
    legend = ax_c.legend(title=None, loc="lower right")
    legend.get_frame().set_alpha(1)

    sns.barplot(data=df, x="Ligand", y="r2", hue="Method", ax=ax_d)
    ax_d.set_xticklabels(
        [DETECTION_DISPLAY_NAMES[label._text] for label in ax_d.get_xticklabels()],
        rotation=45,
    )
    ax_d.set_xlabel("Detection")
    ax_d.set_ylabel("Imputation performance ($R^2$)")
    ax_d.set_title("10% Missing")
    legend = ax_d.legend(title=None, loc="lower right")
    legend.get_frame().set_alpha(1)


def figure_3e(ax, ax_2):
    if UPDATE_CACHE["3e"]:
        runs = 2
        missingnesss = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        data = Zohar().get_detection_signal()

        df = pd.concat(
            run_repeated_imputation(
                data, impute_missing_ms, runs=runs, missingness=missingness
            )
            for missingness in missingnesss
        )
        df.reset_index(inplace=True, drop=True)
        df.to_csv(CACHE_DIR / "3e.csv")
    else:
        df = pd.read_csv(CACHE_DIR / "3e.csv")
    df = df.rename(columns={"Ligand": "Detection"})
    df["Missingness"] = df["Missingness"] * 100

    ax = ax
    df = df[~(df["Detection"].isin(["IgG1", "IgG3"]))]
    sns.lineplot(data=df, x="Missingness", y="r", hue="Detection", ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    labels = [DETECTION_DISPLAY_NAMES[label] for label in labels]
    ax.legend(handles, labels, loc="lower left").get_frame().set_alpha(1)
    ax.set_ylabel("Imputation performance ($r$)")
    ax.set_xlabel("Fraction Missing (%)")
    ax.set_title("Variable missingness")

    ax = ax_2
    sns.lineplot(data=df, x="Missingness", y="r2", hue="Detection", ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    labels = [DETECTION_DISPLAY_NAMES[label] for label in labels]
    ax.legend(handles, labels, loc="lower left").get_frame().set_alpha(1)
    ax.set_ylabel("Imputation performance ($R^2$)")
    ax.set_xlabel("Fraction Missing (%)")
    ax.set_title("Variable missingness")
