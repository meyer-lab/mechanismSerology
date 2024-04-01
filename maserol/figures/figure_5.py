import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import pearsonr
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
    plot = Multiplot(
        (10, 2),
        fig_size=(7.5, 7.5 * 2 / 3),
        subplot_specs=[
            (0, 3, 0, 1),
            (3, 3, 0, 1),
            (6, 4, 0, 1),
            (0, 10, 1, 1),
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

    if UPDATE_CACHE:
        x, ctx = optimize_loss(detection_signal, **opts, return_reshaped_params=True)
        Rtot = Rtot_to_df(x["Rtot"], detection_signal, rcps=list(opts["rcps"]))
        Rtot.to_csv(CACHE_DIR / "zohar_Rtot.csv")
    else:
        Rtot = pd.read_csv(CACHE_DIR / "zohar_Rtot.csv").set_index(
            ["Sample", "Antigen"], drop=True
        )

    fucose_inferred = compute_fucose_ratio(Rtot).xs("S", level="Antigen")
    df_merged = pd.merge(fucose_inferred, metadata, how="inner", on="Sample")

    ax = ax_a
    sns.boxplot(
        data=df_merged,
        x="ARDS",
        y="fucose_inferred",
        ax=ax,
        order=["No", "Yes"],
        palette=sns.color_palette(),
        showfliers=False,
    )
    ax.set_ylabel("anti-S IgG Fucosylation (%)")
    pairs = (("Yes", "No"),)
    annotator = Annotator(ax, pairs, data=df_merged, x="ARDS", y="fucose_inferred")
    annotate_mann_whitney(annotator)

    ax = ax_b
    sns.boxplot(
        data=df_merged,
        x="immunosup",
        y="fucose_inferred",
        ax=ax,
        order=[0, 1],
        showfliers=False,
    )
    # ax.set_ylabel("anti-S IgG Fucosylation (%)")
    ax.set_ylabel(None)
    ax.set_xlabel("Immunosuppressed")
    ax.set_xticklabels(["No", "Yes"])
    pairs = ((1, 0),)
    annotator = Annotator(ax, pairs, data=df_merged, x="immunosup", y="fucose_inferred")
    annotate_mann_whitney(annotator)

    ax = ax_c
    fucose_over_time = pd.merge(
        fucose_inferred, metadata[["days", "patient_ID"]], how="inner", on="Sample"
    )
    fucose_over_time.dropna(inplace=True)
    fucose_over_time = fucose_over_time.groupby("patient_ID").filter(
        lambda x: x["days"].nunique() >= 2
    )
    rmcorr_result = pg.rm_corr(
        data=fucose_over_time, x="days", y="fucose_inferred", subject="patient_ID"
    )
    r, p = rmcorr_result.r.iloc[0], rmcorr_result.pval.iloc[0]
    fucose_over_time_binned = pd.merge(
        fucose_inferred, zohar.get_days_binned(), how="inner", on="Sample"
    )
    sns.lineplot(
        data=fucose_over_time_binned,
        x="days",
        y="fucose_inferred",
        ax=ax,
    )
    ax.text(
        0.7,
        0.8,
        f"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.text(
        0.7,
        0.72,
        r"p = " + "{:.2e}".format(p),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.set_xlabel("Days following symptom onset")
    # ax.set_ylabel("anti-S IgG Fucosylation (%)")
    ax.set_ylabel(None)
    ax.set_xlim(0, 30)


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
        showfliers=False,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize="small")
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
