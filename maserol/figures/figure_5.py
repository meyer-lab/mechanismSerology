import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

from maserol.core import optimize_loss
from maserol.datasets import KaplonekVaccine, Zohar
from maserol.figures.common import CACHE_DIR, Multiplot, annotate_mann_whitney
from maserol.util import Rtot_to_df, assemble_options, compute_fucose_ratio

UPDATE_CACHE = {
    "zohar": False,
    "kaplonek_vaccine": False,
}


def makeFigure():
    plot = Multiplot(
        (5, 2),
        fig_size=(7.5, 7.5 * 2 / 3),
        subplot_specs=[
            (0, 2, 0, 1),
            (2, 1, 0, 1),
            (3, 2, 0, 1),
            (0, 5, 1, 1),
        ],
    )
    figure_5abc(plot.axes[0], plot.axes[1], plot.axes[2])
    figure_5d(plot.axes[3])
    plot.add_subplot_labels()
    plot.fig.tight_layout(pad=0, w_pad=0.2, h_pad=1)
    return plot.fig


def figure_5abc(ax_a, ax_b, ax_c):
    zohar = Zohar()
    detection_signal = zohar.get_detection_signal()
    metadata = zohar.get_metadata()

    opts = assemble_options(detection_signal)

    if UPDATE_CACHE["zohar"]:
        x, ctx = optimize_loss(detection_signal, **opts, return_reshaped_params=True)
        Rtot = Rtot_to_df(x["Rtot"], detection_signal, rcps=list(opts["rcps"]))
        Rtot.to_csv(CACHE_DIR / "zohar_Rtot.csv")
    else:
        Rtot = pd.read_csv(CACHE_DIR / "zohar_Rtot.csv").set_index(
            ["Sample", "Antigen"], drop=True
        )

    fucose_inferred = compute_fucose_ratio(Rtot).reset_index(level="Antigen")

    y_lim = (26, 103)

    ax = ax_a
    order = ["N", "S", "S2", "S1", "S1 Trimer", "RBD"]
    sns.boxplot(
        data=fucose_inferred,
        x="Antigen",
        y="fucose_inferred",
        ax=ax,
        order=order,
        showfliers=False,
        palette=["#539ecd"],
    )
    ax.set_ylabel("IgG Fucosylation (%)")
    ax.set_ylim(*y_lim)
    pairs = (("S2", "S1"), ("N", "S"))
    annotator = Annotator(
        ax, pairs, data=fucose_inferred, x="Antigen", y="fucose_inferred", order=order
    )
    annotate_mann_whitney(annotator)

    ax = ax_b
    fucose_inferred["Antigen type"] = [
        "N" if "N" in ag else "S-associated" for ag in fucose_inferred.Antigen
    ]
    sns.boxplot(
        data=fucose_inferred,
        x="Antigen type",
        y="fucose_inferred",
        ax=ax,
        showfliers=False,
        palette=["#61bdcd"],
        saturation=1,
    )
    ax.set_ylabel(None)
    ax.set_ylim(*y_lim)
    ax.set_yticklabels([])
    pairs = (("N", "S-associated"),)
    annotator = Annotator(
        ax, pairs, data=fucose_inferred, x="Antigen type", y="fucose_inferred"
    )
    annotate_mann_whitney(annotator)

    ax = ax_c
    df_merged = pd.merge(fucose_inferred, metadata["ARDS"], how="inner", on="Sample")
    sns.boxplot(
        data=df_merged,
        x="Antigen",
        hue="ARDS",
        y="fucose_inferred",
        ax=ax,
        hue_order=["No", "Yes"],
        palette=["#bad6eb", "#0b559f"],
        showfliers=False,
        order=order,
    )
    ax.set_ylabel(None)
    ax.set_ylim(*y_lim)
    ax.set_yticklabels([])
    pairs = [((ag, "No"), (ag, "Yes")) for ag in df_merged.Antigen.unique()]
    # set legend items to "No ARDS" and "ARDS"
    ax.legend_.set_title(None)
    ax.legend_.texts[0].set_text("non-ARDS")
    ax.legend_.texts[1].set_text("ARDS")
    annotator = Annotator(
        ax, pairs, data=df_merged, x="Antigen", hue="ARDS", y="fucose_inferred"
    )
    annotate_mann_whitney(annotator)


def figure_5d(ax):
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
    filepath = CACHE_DIR / "kaplonek_vaccine_Rtot.csv"
    if UPDATE_CACHE["kaplonek_vaccine"]:
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
        showfliers=False,
        palette=["#a9aa35", "#c03d3e"],
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_ylabel("IgG Fucosylation (%)")
    ax.set_ylim(-2, 102)
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

    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["Avoided infection", "Infected"]
    legend = ax.legend(handles, new_labels, title=None, loc="lower right")
    legend.get_frame().set_alpha(1)
