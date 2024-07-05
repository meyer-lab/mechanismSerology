import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from maserol.core import optimize_loss
from maserol.datasets import Alter
from maserol.figures.common import (
    ANNOTATION_FONT_SIZE,
    DETECTION_DISPLAY_NAMES,
    Multiplot,
)
from maserol.figures.figure_4 import ALTER_RTOT_CACHE_PATH
from maserol.util import (
    IgG1_3,
    Rtot_to_df,
    assemble_options,
    compute_fucose_ratio,
    data_to_df,
)

ALPHA = 0.65
POINT_SIZE = 12
ANNOTATION_LOCATION = (0.07, 0.85)
ANNOTATION_SEPARATION = 0.06
CORRELATION_SYMBOL = r"$\mathrm{r_S}$"

UPDATE_CACHE = False


def makeFigure():
    """Relies on result from figure 4"""
    function = Alter().get_effector_functions()
    detection_signal = data_to_df(Alter().get_detection_signal()).xs(
        "gp120.SF162", level="Antigen"
    )
    R3A_R2A = np.log10(
        detection_signal["FcR3A-158V"] / detection_signal["FcR2A-131H"] + 1
    )
    R3A_R2A.name = "R3A/R2A"
    R3A_IgG = np.log10(detection_signal["FcR3A-158V"] / detection_signal["IgG1"] + 1)
    R3A_IgG.name = "R3A/IgG1"
    fucose_ce = Alter().get_fucose_data()
    Rtot = pd.read_csv(ALTER_RTOT_CACHE_PATH)
    Rtot.set_index(["Sample", "Antigen"], inplace=True)
    fucose_inferred = compute_fucose_ratio(Rtot).xs("gp120.SF162", level="Antigen")
    plot = Multiplot(
        (3, 4),
        fig_size=(7.5, 8.75),
        subplot_specs=[
            (0, 2, 0, 1),
            (2, 1, 0, 1),
            (0, 1, 1, 1),
            (1, 1, 1, 1),
            (2, 1, 1, 1),
            (0, 1, 2, 1),
            (1, 1, 2, 1),
            (2, 1, 2, 1),
            (0, 1, 3, 1),
            (1, 1, 3, 1),
            (2, 1, 3, 1),
        ],
    )

    figure_CE(plot.axes[1])

    ax = plot.axes[2]
    df = pd.merge(R3A_IgG, fucose_ce, how="inner", on="Sample")
    annotate_spearman(ax, df["fucose_ce"], df["R3A/IgG1"])
    sns.scatterplot(
        data=df, x="fucose_ce", y="R3A/IgG1", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    ax.set_ylabel(
        r"$\mathrm{log_{10}}$"
        f"({DETECTION_DISPLAY_NAMES['FcR3A']}"
        f" / {DETECTION_DISPLAY_NAMES['IgG1']} + 1)"
    )
    ax.set_xlabel("CE IgG Fucosylation (%)")

    ax = plot.axes[3]
    df = pd.merge(R3A_R2A, fucose_ce, how="inner", on="Sample")
    annotate_spearman(ax, df["fucose_ce"], df["R3A/R2A"])
    sns.scatterplot(
        data=df, x="fucose_ce", y="R3A/R2A", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    ax.set_ylabel(
        r"$\mathrm{log_{10}}$"
        f"({DETECTION_DISPLAY_NAMES['FcR3A']}"
        f" / {DETECTION_DISPLAY_NAMES['FcR2A']} + 1)"
    )
    ax.set_xlabel("CE IgG Fucosylation (%)")

    ax = plot.axes[4]
    df = pd.merge(R3A_IgG, fucose_inferred, how="inner", on="Sample")
    annotate_spearman(ax, df["fucose_inferred"], df["R3A/IgG1"])
    sns.scatterplot(
        data=df, x="fucose_inferred", y="R3A/IgG1", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    ax.set_ylabel(
        r"$\mathrm{log_{10}}$"
        f"({DETECTION_DISPLAY_NAMES['FcR3A']}"
        f" / {DETECTION_DISPLAY_NAMES['IgG1']} + 1)"
    )
    ax.set_xlabel("Inferred IgG Fucosylation (%)")

    ax = plot.axes[5]
    df = pd.merge(R3A_R2A, fucose_inferred, how="inner", on="Sample")
    annotate_spearman(ax, df["fucose_inferred"], df["R3A/R2A"])
    sns.scatterplot(
        data=df, x="fucose_inferred", y="R3A/R2A", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    ax.set_ylabel(
        r"$\mathrm{log_{10}}$"
        f"({DETECTION_DISPLAY_NAMES['FcR3A']}"
        f" / {DETECTION_DISPLAY_NAMES['FcR2A']} + 1)"
    )
    ax.set_xlabel("Inferred IgG Fucosylation (%)")

    ax = plot.axes[6]
    df = pd.merge(
        detection_signal["FcR3A-158V"], function["MIP1b"], how="inner", on="Sample"
    ).dropna()
    annotate_spearman(ax, df["FcR3A-158V"], df["MIP1b"])
    sns.scatterplot(
        data=df, x="FcR3A-158V", y="MIP1b", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    ax.set_ylabel("ADNKA")
    ax.set_xlabel(f"{DETECTION_DISPLAY_NAMES['FcR3A']}")

    ax = plot.axes[7]
    df = pd.merge(
        detection_signal["FcR2A-131H"], function["ADNP"], how="inner", on="Sample"
    ).dropna()
    annotate_spearman(ax, df["FcR2A-131H"], df["ADNP"])
    sns.scatterplot(data=df, x="FcR2A-131H", y="ADNP", ax=ax, alpha=ALPHA, s=POINT_SIZE)
    ax.set_ylabel("ADNP")
    ax.set_xlabel(f"{DETECTION_DISPLAY_NAMES['FcR2A']}")

    ax = plot.axes[8]
    glycan_ce = Alter().get_glycan_data()
    glycan_ce = glycan_ce[glycan_ce["F.total"] > 0]
    sns.scatterplot(
        data=glycan_ce, x="F.total", y="B.total", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    ax.set_ylabel("CE Bisecting GlcNAc (%)")
    ax.set_xlabel("CE Fucosylation (%)")
    annotate_spearman(ax, glycan_ce["F.total"], glycan_ce["B.total"])

    plot.add_subplot_labels(ax_relative=True)
    plot.fig.tight_layout(pad=0, w_pad=2, h_pad=0.3)
    return plot.fig


def figure_CE(ax):
    fucose_ce = Alter().get_fucose_data()
    if UPDATE_CACHE:
        detection_signal = Alter().get_detection_signal()
        opts = assemble_options(detection_signal, rcps=IgG1_3)
        params, _ = optimize_loss(detection_signal, **opts, return_reshaped_params=True)
        Rtot = Rtot_to_df(params["Rtot"], data=detection_signal, rcps=list(IgG1_3))
        Rtot.to_csv(ALTER_RTOT_CACHE_PATH)
    else:
        Rtot = pd.read_csv(ALTER_RTOT_CACHE_PATH)
        Rtot.set_index(["Sample", "Antigen"], inplace=True)
    fucose_inferred = compute_fucose_ratio(Rtot).xs("gp120.SF162", level="Antigen")
    fucose_compare = pd.merge(
        left=fucose_inferred, right=fucose_ce, on="Sample", how="inner"
    )

    sns.scatterplot(
        data=fucose_compare,
        y="fucose_ce",
        x="fucose_inferred",
        ax=ax,
        alpha=ALPHA,
        s=POINT_SIZE,
    )
    ax.set_ylabel("CE IgG Fucosylation (%)")
    ax.set_xlabel("Inferred IgG Fucosylation (%)")
    r, p = pearsonr(fucose_compare["fucose_ce"], fucose_compare["fucose_inferred"])
    ax.set_title("Model Inferences vs CE Measurements")
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1],
        r"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
    )
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1] - ANNOTATION_SEPARATION,
        r"p = " + f"{p:.2e}",
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lims = [
        min(xlim[0], ylim[0]),
        max(xlim[1], ylim[1]),
    ]
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.plot(lims, lims, linestyle="--", color="gray", alpha=0.75)


def annotate_spearman(ax, x, y):
    r, p = spearmanr(x, y)
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1],
        f"{CORRELATION_SYMBOL} = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
    )
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1] - ANNOTATION_SEPARATION,
        r"p = " + f"{p:.2e}",
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
    )
