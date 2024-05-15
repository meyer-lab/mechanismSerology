import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from maserol.core import optimize_loss
from maserol.datasets import Alter, Zohar
from maserol.figures.common import (CACHE_DIR, DETECTION_DISPLAY_NAMES,
                                    Multiplot)
from maserol.util import (IgG1_3, Rtot_to_df, assemble_options,
                          compute_fucose_ratio)

ALPHA = 0.72
UPDATE_CACHE = {
    "zohar": False,
    "alter": False,
}
ALTER_RTOT_CACHE_PATH = CACHE_DIR / "alter_Rtot.csv"


def makeFigure():
    plot = Multiplot(
        (3, 2),
        fig_size=(7.5, 7.5 * 2 / 3 + 0.1),
        subplot_specs=[
            (0, 1, 0, 1),
            (1, 1, 0, 1),
            (2, 1, 0, 1),
            (0, 2, 1, 1),
            (2, 1, 1, 1),
        ],
    )
    figure_mechanistic_relate(plot.axes[0], plot.axes[1], plot.axes[2])
    figure_CE(plot.axes[4])
    plot.add_subplot_labels(ax_relative=True)
    plot.fig.tight_layout(pad=0, w_pad=0.2, h_pad=1)
    return plot.fig


def figure_CE(ax):
    fucose_ce = Alter().get_fucose_data()
    if UPDATE_CACHE["alter"]:
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
        data=fucose_compare, y="fucose_ce", x="fucose_inferred", ax=ax, alpha=ALPHA
    )
    ax.set_ylabel("CE IgG Fucosylation (%)")
    ax.set_xlabel("Inferred IgG Fucosylation (%)")
    r, p = pearsonr(fucose_compare["fucose_ce"], fucose_compare["fucose_inferred"])
    ax.set_title("Model Inferences vs CE Measurements")
    ax.text(
        0.6,
        0.06,
        r"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.text(
        0.6,
        0.01,
        r"p = " + "{:.2e}".format(p),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
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


def figure_mechanistic_relate(ax_0, ax_1, ax_2):
    zohar = Zohar()
    if UPDATE_CACHE["zohar"]:
        detection_signal = zohar.get_detection_signal()
        opts = assemble_options(detection_signal)
        x, ctx = optimize_loss(detection_signal, **opts, return_reshaped_params=True)
        Rtot = Rtot_to_df(x["Rtot"], detection_signal, rcps=list(opts["rcps"]))
        Rtot.to_csv(CACHE_DIR / "zohar_Rtot.csv")
    else:
        Rtot = pd.read_csv(CACHE_DIR / "zohar_Rtot.csv").set_index(
            ["Sample", "Antigen"], drop=True
        )
    Rtot = Rtot.xs("S", level="Antigen")

    fucose = compute_fucose_ratio(Rtot)
    df = zohar.get_metadata()
    df = pd.merge(df, fucose, on="Sample", how="inner")

    y = np.log10(df["FcR3A_S"] / df["FcR2A_S"])
    sns.scatterplot(
        y=y,
        x=df["fucose_inferred"],
        ax=ax_0,
        alpha=ALPHA,
    )
    ax_0.set_xlabel("Inferred IgG Fucosylation (%)")
    ax_0.set_ylabel(
        r"$\mathrm{log_{10}}$"
        + f"({DETECTION_DISPLAY_NAMES['FcR3A']} / {DETECTION_DISPLAY_NAMES['FcR2A']})"
    )
    r, p = pearsonr(df["fucose_inferred"], y)
    ax_0.text(
        0.05,
        0.06,
        r"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax_0.transAxes,
    )

    sns.scatterplot(
        data=df, y="ADNKA_CD107a_S", x="fucose_inferred", ax=ax_1, alpha=ALPHA
    )
    ax_1.set_xlabel("Inferred IgG Fucosylation (%)")
    ax_1.set_ylabel(r"ADCC (CD107a)")
    r, p = pearsonr(df["fucose_inferred"], df["ADNKA_CD107a_S"])
    ax_1.text(
        0.05,
        0.86,
        r"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax_1.transAxes,
    )

    sns.scatterplot(
        data=df, y="ADNKA_MIP1b_S", x="fucose_inferred", ax=ax_2, alpha=ALPHA
    )
    ax_2.set_xlabel("Inferred IgG Fucosylation (%)")
    ax_2.set_ylabel(r"ADCC (MIP1b)")
    r, p = pearsonr(df["fucose_inferred"], df["ADNKA_MIP1b_S"])
    ax_2.text(
        0.05,
        0.86,
        r"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax_2.transAxes,
    )
