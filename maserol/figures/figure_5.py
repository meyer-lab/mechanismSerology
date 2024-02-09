import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

from maserol.core import optimize_loss
from maserol.datasets import Zohar, KaplonekVaccine
from maserol.figures.common import (
    Multiplot,
    CACHE_DIR,
    annotate_mann_whitney,
)
from maserol.util import (
    assemble_options,
    Rtot_to_df,
    compute_fucose_ratio,
)

UPDATE_CACHE = False


def makeFigure():
    plot = Multiplot((3, 2.5), (3, 2), multz={3: 2})
    figure_5a(plot.axes[0])
    figure_5b(plot.axes[3])
    plot.add_subplot_labels()
    plot.fig.tight_layout()
    return plot.fig


def figure_5a(ax):
    zohar = Zohar()
    detection_signal = zohar.get_detection_signal()
    ARDS = zohar.get_ARDS()

    opts = assemble_options(detection_signal)

    if UPDATE_CACHE:
        x, ctx = optimize_loss(detection_signal, **opts, return_reshaped_params=True)
        Rtot = Rtot_to_df(x["Rtot"], detection_signal, rcps=list(opts["rcps"]))
        Rtot.to_csv(CACHE_DIR / "fig_5a_Rtot.csv")
    else:
        Rtot = pd.read_csv(CACHE_DIR / "fig_5a_Rtot.csv").set_index(
            ["Sample", "Antigen"], drop=True
        )

    fucose_inferred = compute_fucose_ratio(Rtot).xs("S", level="Antigen")

    df_merged = pd.merge(fucose_inferred, ARDS, how="inner", on="Sample")
    df_merged.sort_values("ARDS", inplace=True)

    sns.boxplot(data=df_merged, x="ARDS", y="fucose_inferred", ax=ax)
    ax.set_xlabel(None)
    ax.set_xticklabels(["Non-ARDS", "ARDS"])
    ax.set_ylabel("anti-S IgG Fucosylation (%)")

    pairs = (("Yes", "No"),)
    annotator = Annotator(ax, pairs, data=df_merged, x="ARDS", y="fucose_inferred")
    annotate_mann_whitney(annotator)


def figure_5b(ax):
    AG_EXCLUDE = [
        "Ebola",
        "HKU1.Spike",
        "Influenza",
        "CMV",
        "OC43.Spike",
        "P1.RBD",
        "P1.S",
        "RSV",
        "CMV",
    ]

    kaplonek_vaccine = KaplonekVaccine()
    detection_signal = kaplonek_vaccine.get_detection_signal()
    ag_include = [
        ag for ag in np.unique(detection_signal.Antigen.values) if ag not in AG_EXCLUDE
    ]
    detection_signal = detection_signal.sel(Complex=pd.IndexSlice[:, ag_include])
    metadata = kaplonek_vaccine.get_metadata()
    filepath = CACHE_DIR / "fig_5b_Rtot.csv"
    if UPDATE_CACHE:
        opts = assemble_options(detection_signal)
        params, _ = optimize_loss(detection_signal, **opts, return_reshaped_params=True)
        Rtot = Rtot_to_df(params["Rtot"], detection_signal, list(opts["rcps"]))
        Rtot.to_csv(filepath)
    else:
        Rtot = pd.read_csv(filepath)
        Rtot.set_index(["Sample", "Antigen"], inplace=True, drop=True)

    fucose_inferred = compute_fucose_ratio(Rtot).reset_index(level="Antigen")
    df_compare = pd.merge(
        fucose_inferred, metadata["infection.status"], how="inner", on="Sample"
    ).reset_index()
    df_compare = df_compare.sort_values("Antigen")[::-1]

    sns.boxplot(
        data=df_compare,
        x="Antigen",
        y="fucose_inferred",
        hue="infection.status",
        ax=ax,
        hue_order=["control", "case"],
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel("IgG Fucosylation (%)")
    ax.set_xlabel("Antigen")
    pairs = [((ag, "control"), (ag, "case")) for ag in df_compare.Antigen.unique()]
    annotator = Annotator(
        ax,
        pairs,
        data=df_compare,
        x="Antigen",
        y="fucose_inferred",
        hue="infection.status",
    )
    annotate_mann_whitney(annotator)

    ax.legend(title=None)
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["Healthy", "COVID-19$^+$"]
    ax.legend(handles, new_labels)

    sns.move_legend(ax, "lower right")
