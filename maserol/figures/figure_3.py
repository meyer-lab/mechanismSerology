import functools

import numpy as np
import pandas as pd
import seaborn as sns

from maserol.datasets import Zohar
from maserol.figures.common import (
    CACHE_DIR,
    DETECTION_DISPLAY_NAMES,
    LOG10_SYMBOL,
    Multiplot,
)
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
R_FIGURE_RANGE = [0, 1.02]
R2_FIGURE_RANGE = [-0.55, 1.02]


def makeFigure():
    plot = Multiplot(
        (9, 3),
        fig_size=(7.5, 7.7),
        subplot_specs=[
            (0, 3, 0, 1),
            (3, 6, 0, 1),
            (0, 3, 1, 1),
            (3, 3, 1, 1),
            (6, 3, 1, 1),
            (0, 3, 2, 1),
            (3, 3, 2, 1),
            (6, 3, 2, 1),
        ],
    )
    figure_imputation_scatterplot(plot.axes[2])
    figure_compare_pca(plot.axes[3], plot.axes[4])
    figure_variable_missingness(plot.axes[6], plot.axes[7])
    plot.fig.tight_layout(pad=0, w_pad=-0.5, h_pad=-0.5)
    plot.add_subplot_labels(ax_relative=True)
    return plot.fig


def figure_imputation_scatterplot(ax):
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

    ax.set_xticks([1e2, 1e4, 1e6])
    ax.set_yticks([1e2, 1e4, 1e6])
    ax.set_xlabel(f"Inferred {DETECTION_DISPLAY_NAMES["FcR3B"]} (RFU)")
    ax.set_ylabel(f"Measured {DETECTION_DISPLAY_NAMES["FcR3B"]} (RFU)")
    ax.set_title("10% Missing")
    lim = [3, 1e7]
    ax.set_ylim(lim)
    ax.set_xlim(lim)
    ax.plot(lim, lim, linestyle="--", color="gray", alpha=0.75)


def figure_compare_pca(ax_r, ax_r2):
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

    sns.barplot(data=df, x="Ligand", y="r", hue="Method", ax=ax_r)
    ax_r.set_xticklabels(
        [DETECTION_DISPLAY_NAMES[label._text] for label in ax_r.get_xticklabels()],
        rotation=45,
    )
    ax_r.set_xlabel("Detection")
    ax_r.set_ylabel("Imputation performance ($r$)")
    ax_r.set_title("10% Missing")
    ax_r.set_ylim(R_FIGURE_RANGE)
    legend = ax_r.legend(title=None, loc="lower right")
    legend.get_frame().set_alpha(1)

    sns.barplot(data=df, x="Ligand", y="r2", hue="Method", ax=ax_r2)
    ax_r2.set_xticklabels(
        [DETECTION_DISPLAY_NAMES[label._text] for label in ax_r2.get_xticklabels()],
        rotation=45,
    )
    ax_r2.set_xlabel("Detection")
    ax_r2.set_ylabel("Imputation performance ($R^2$)")
    ax_r2.set_title("10% Missing")
    ax_r2.set_ylim(R2_FIGURE_RANGE)
    legend = ax_r2.legend(title=None, loc="lower right")
    legend.get_frame().set_alpha(1)


def figure_variable_missingness(ax_r, ax_r2):
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

    ax = ax_r
    df = df[~(df["Detection"].isin(["IgG1", "IgG3"]))]
    sns.lineplot(data=df, x="Missingness", y="r", hue="Detection", ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    labels = [DETECTION_DISPLAY_NAMES[label] for label in labels]
    ax.legend(handles, labels, loc="upper right").get_frame().set_alpha(1)
    ax.set_ylabel("Imputation performance ($r$)")
    ax.set_xlabel("Fraction Missing (%)")
    ax.set_title("Variable missingness")
    ax.set_ylim(R_FIGURE_RANGE)

    ax = ax_r2
    sns.lineplot(data=df, x="Missingness", y="r2", hue="Detection", ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    labels = [DETECTION_DISPLAY_NAMES[label] for label in labels]
    ax.legend(handles, labels, loc="lower left").get_frame().set_alpha(1)
    ax.set_ylabel("Imputation performance ($R^2$)")
    ax.set_xlabel("Fraction Missing (%)")
    ax.set_title("Variable missingness")
    ax.set_ylim(R2_FIGURE_RANGE)
