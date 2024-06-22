import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from maserol.core import optimize_loss
from maserol.datasets import Zohar
from maserol.figures.common import (
    ANNOTATION_FONT_SIZE,
    CACHE_DIR,
    DETECTION_DISPLAY_NAMES,
    LOG10_SYMBOL,
    Multiplot,
)
from maserol.util import Rtot_to_df, assemble_options, compute_fucose_ratio

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
            (1, 2, 0, 1),
            (0, 1, 1, 1),
            (1, 1, 1, 1),
            (2, 1, 1, 1),
        ],
    )
    figure_mechanistic_relate(plot.axes[2], plot.axes[3], plot.axes[4])
    plot.add_subplot_labels(ax_relative=True)
    plot.fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    return plot.fig


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
    df_meta = zohar.get_metadata()
    df_fucose_ratio = pd.merge(df_meta, fucose, on="Sample", how="inner")
    df_Rtot = pd.merge(Rtot, df_meta, on="Sample", how="inner")
    df_Rtot["total_afucosylated"] = df_Rtot["IgG1"] + df_Rtot["IgG3"]

    y = np.log10(df_fucose_ratio["FcR3A_S"] / df_fucose_ratio["FcR2A_S"] + 1)
    sns.scatterplot(
        y=y,
        x=df_fucose_ratio["fucose_inferred"],
        ax=ax_0,
        alpha=ALPHA,
    )
    ax_0.set_xlabel("Inferred IgG Fucosylation (%)")
    ax_0.set_ylabel(
        f"{LOG10_SYMBOL}({DETECTION_DISPLAY_NAMES['FcR3A']} / {DETECTION_DISPLAY_NAMES['FcR2A']} + 1)"
    )
    r, p = pearsonr(df_fucose_ratio["fucose_inferred"], y)
    ax_0.text(
        0.05,
        0.05,
        r"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax_0.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
    )

    afucosylated_xlim = (-0.1e7, 1.4e7)
    afucosylated_xticks = np.linspace(0, 1.2e7, 4)
    afucosylated_xticklabels = [
        "0",
        r"4$\times 10^6$",
        r"8$\times 10^6$",
        r"12$\times 10^6$",
    ]

    sns.scatterplot(
        data=df_Rtot,
        y="ADNKA_CD107a_S",
        x="total_afucosylated",
        ax=ax_1,
        alpha=ALPHA,
    )
    ax_1.set_xlabel("Inferred afucosylated IgG")
    ax_1.set_ylabel(r"ADCC (CD107a)")
    ax_1.set_xticks(afucosylated_xticks)
    ax_1.set_xticklabels(afucosylated_xticklabels)
    ax_1.set_xlim(afucosylated_xlim)
    r, p = pearsonr(df_Rtot["total_afucosylated"], df_Rtot["ADNKA_CD107a_S"])
    ax_1.text(
        0.05,
        0.86,
        r"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax_1.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
    )

    sns.scatterplot(
        data=df_Rtot,
        y="ADNKA_MIP1b_S",
        x="total_afucosylated",
        ax=ax_2,
        alpha=ALPHA,
    )
    ax_2.set_xlabel("Inferred afucosylated IgG")
    ax_2.set_ylabel(r"ADCC (MIP1b)")
    ax_2.set_xticks(afucosylated_xticks)
    ax_2.set_xlim(afucosylated_xlim)
    ax_2.set_xticklabels(afucosylated_xticklabels)
    r, p = pearsonr(df_Rtot["total_afucosylated"], df_Rtot["ADNKA_MIP1b_S"])
    ax_2.text(
        0.05,
        0.86,
        r"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax_2.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
    )
