import pandas as pd
import numpy as np
import seaborn as sns

from maserol.core import optimize_loss
from maserol.util import assemble_options, Rtot_to_df, compute_fucose_ratio, data_to_df
from maserol.datasets import Zohar, Alter
from maserol.figures.common import Multiplot, DETECTION_DISPLAY_NAMES, CACHE_DIR
from maserol.figures.figure_4 import ALTER_RTOT_CACHE_PATH
from scipy.stats import spearmanr, pearsonr

ALPHA = 0.65
POINT_SIZE = 12
ANNOTATION_LOCATION = (0.07, 0.83)
ANNOTATION_SEPARATION = 0.06
CORRELATION_SYMBOL = "$\mathrm{r_S}$"


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
    ADNKA_ADNP = function["MIP1b"] / function["ADNP"]
    ADNKA_ADNP.name = "ADNKA/ADNP"
    fucose_ce = Alter().get_fucose_data()
    Rtot = pd.read_csv(ALTER_RTOT_CACHE_PATH)
    Rtot.set_index(["Sample", "Antigen"], inplace=True)
    fucose_inferred = compute_fucose_ratio(Rtot).xs("gp120.SF162", level="Antigen")
    plot = Multiplot((3, 3), fig_size=(7.5, 7.3))

    ax = plot.axes[0]
    df = pd.merge(R3A_IgG, fucose_ce, how="inner", on="Sample")
    r, p = spearmanr(df["fucose_ce"], df["R3A/IgG1"])
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1],
        f"{CORRELATION_SYMBOL} = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1] - ANNOTATION_SEPARATION,
        r"p = " + str(round(p, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    sns.scatterplot(
        data=df, x="fucose_ce", y="R3A/IgG1", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    ax.set_ylabel(
        r"$\mathrm{log_{10}}$"
        + f"({DETECTION_DISPLAY_NAMES['FcR3A']} / {DETECTION_DISPLAY_NAMES['IgG1']})"
    )
    ax.set_xlabel("CE IgG Fucosylation (%)")

    ax = plot.axes[3]
    df = pd.merge(R3A_R2A, fucose_ce, how="inner", on="Sample")
    r, p = spearmanr(df["fucose_ce"], df["R3A/R2A"])
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1],
        f"{CORRELATION_SYMBOL} = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1] - ANNOTATION_SEPARATION,
        r"p = " + str(round(p, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    sns.scatterplot(
        data=df, x="fucose_ce", y="R3A/R2A", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    ax.set_ylabel(
        r"$\mathrm{log_{10}}$"
        + f"({DETECTION_DISPLAY_NAMES['FcR3A']} / {DETECTION_DISPLAY_NAMES['FcR2A']})"
    )
    ax.set_xlabel("CE IgG Fucosylation (%)")

    ax = plot.axes[1]
    df = pd.merge(R3A_IgG, fucose_inferred, how="inner", on="Sample")
    r, p = spearmanr(df["fucose_inferred"], df["R3A/IgG1"])
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1],
        f"{CORRELATION_SYMBOL} = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1] - ANNOTATION_SEPARATION,
        r"p = " + "{:.2e}".format(p),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    sns.scatterplot(
        data=df, x="fucose_inferred", y="R3A/IgG1", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    ax.set_ylabel(
        r"$\mathrm{log_{10}}$"
        + f"({DETECTION_DISPLAY_NAMES['FcR3A']} / {DETECTION_DISPLAY_NAMES['IgG1']})"
    )
    ax.set_xlabel("Inferred IgG Fucosylation (%)")

    ax = plot.axes[4]
    df = pd.merge(R3A_R2A, fucose_inferred, how="inner", on="Sample")
    r, p = spearmanr(df["fucose_inferred"], df["R3A/R2A"])
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1],
        f"{CORRELATION_SYMBOL} = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1] - ANNOTATION_SEPARATION,
        r"p = " + str(round(p, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    sns.scatterplot(
        data=df, x="fucose_inferred", y="R3A/R2A", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    ax.set_ylabel(
        r"$\mathrm{log_{10}}$"
        + f"({DETECTION_DISPLAY_NAMES['FcR3A']} / {DETECTION_DISPLAY_NAMES['FcR2A']})"
    )
    ax.set_xlabel("Inferred IgG Fucosylation (%)")

    ax = plot.axes[2]
    df = pd.merge(
        detection_signal["FcR3A-158V"], function["MIP1b"], how="inner", on="Sample"
    ).dropna()
    r, p = spearmanr(df["FcR3A-158V"], df["MIP1b"])
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1],
        f"{CORRELATION_SYMBOL} = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1] - ANNOTATION_SEPARATION,
        r"p = " + "{:.2e}".format(p),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    sns.scatterplot(
        data=df, x="FcR3A-158V", y="MIP1b", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    ax.set_ylabel("ADNKA")
    ax.set_xlabel(f"{DETECTION_DISPLAY_NAMES['FcR3A']}")

    ax = plot.axes[5]
    df = pd.merge(
        detection_signal["FcR2A-131H"], function["ADNP"], how="inner", on="Sample"
    ).dropna()
    r, p = spearmanr(df["FcR2A-131H"], df["ADNP"])
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1],
        f"{CORRELATION_SYMBOL} = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1] - ANNOTATION_SEPARATION,
        r"p = " + "{:.2e}".format(p),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    sns.scatterplot(data=df, x="FcR2A-131H", y="ADNP", ax=ax, alpha=ALPHA, s=POINT_SIZE)
    ax.set_ylabel("ADNP")
    ax.set_xlabel(f"{DETECTION_DISPLAY_NAMES['FcR2A']}")

    ax = plot.axes[6]
    glycan_ce = Alter().get_glycan_data()
    glycan_ce = glycan_ce[glycan_ce["F.total"] > 0]
    sns.scatterplot(
        data=glycan_ce, x="F.total", y="B.total", ax=ax, alpha=ALPHA, s=POINT_SIZE
    )
    # print correlation
    r, p = spearmanr(glycan_ce["F.total"], glycan_ce["B.total"])
    ax.set_ylabel("CE Bisecting GlcNAc (%)")
    ax.set_xlabel("CE Fucosylation (%)")
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1],
        f"{CORRELATION_SYMBOL} = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.text(
        ANNOTATION_LOCATION[0],
        ANNOTATION_LOCATION[1] - ANNOTATION_SEPARATION,
        r"p = " + "{:.2e}".format(p),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    plot.add_subplot_labels()
    return plot.fig
