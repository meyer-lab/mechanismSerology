import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

from maserol.core import optimize_loss
from maserol.datasets import Alter
from maserol.figures.common import (
    Multiplot,
    CACHE_DIR,
    annotate_mann_whitney,
)
from maserol.util import assemble_options, Rtot_to_df, IgG1_3, compute_fucose_ratio

UPDATE_CACHE = False


def makeFigure():
    alter = Alter()
    detection_signal = alter.get_detection_signal()
    subject_class = alter.get_subject_class()

    opts = assemble_options(detection_signal)
    if UPDATE_CACHE:
        params, _ = optimize_loss(detection_signal, **opts, return_reshaped_params=True)
        Rtot = Rtot_to_df(params["Rtot"], data=detection_signal, rcps=list(IgG1_3))
        Rtot.to_csv(CACHE_DIR / "fig_6_Rtot.csv")
    else:
        Rtot = pd.read_csv(CACHE_DIR / "fig_6_Rtot.csv")
        Rtot.set_index(["Sample", "Antigen"], inplace=True)

    fucose_inferred = compute_fucose_ratio(Rtot).reset_index(level="Antigen")
    df_compare = pd.merge(
        fucose_inferred, subject_class, how="inner", on="Sample"
    ).reset_index()

    plot = Multiplot(
        (3, 2.5),
        (3, 2),
        subplot_specs=[
            (0, 3, 0, 1),
            (0, 2, 1, 1),
            (2, 1, 1, 1),
        ],
    )

    # a
    ax = plot.axes[0]
    sns.boxplot(data=df_compare, x="Antigen", y="fucose_inferred", hue="class", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontsize="small")
    ax.set_xlabel("Antigen", labelpad=0)
    ax.set_ylabel("IgG Fucosylation (%)")
    ax.legend(title=None)
    sns.move_legend(ax, "lower right")

    # b
    ax = plot.axes[1]
    df_compare["is_EC"] = df_compare["class"] == "EC"
    sns.boxplot(
        data=df_compare,
        x="Antigen",
        y="fucose_inferred",
        hue="is_EC",
        ax=ax,
        hue_order=[True, False],
    )
    ax.set_xlabel("Antigen", labelpad=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontsize="small")
    ax.set_ylabel("IgG Fucosylation (%)")
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["EC", "Others"]
    ax.legend(handles, new_labels)
    sns.move_legend(ax, "lower right")
    pairs = [((ag, False), (ag, True)) for ag in df_compare.Antigen.unique()]
    annotator = Annotator(
        ax, pairs, data=df_compare, x="Antigen", y="fucose_inferred", hue="is_EC"
    )
    annotate_mann_whitney(annotator)

    # c
    ax = plot.axes[2]
    df_compare["Antigen Category"] = np.ones(len(df_compare))
    for i in df_compare.index:
        antigen = df_compare.loc[i, "Antigen"]
        if (
            antigen.startswith("gp120")
            or antigen.startswith("gp140")
            or antigen.startswith("gp41")
            or antigen == "SOSIP"
        ):
            df_compare.loc[i, "Antigen Category"] = "Env trimer"
        elif antigen.startswith("p24"):
            df_compare.loc[i, "Antigen Category"] = "p24"
        else:
            df_compare.loc[i, "Antigen Category"] = antigen

    sns.boxplot(data=df_compare, x="Antigen Category", y="fucose_inferred", ax=ax)
    ax.set_ylabel("IgG Fucosylation (%)")
    ax.set_xlabel("Antigen type")

    pairs = [("Env trimer", "p24"), ("Env trimer", "IIIb.pr55.Gag")]
    annotator = Annotator(
        ax, pairs, data=df_compare, x="Antigen Category", y="fucose_inferred"
    )
    annotate_mann_whitney(annotator)

    plot.add_subplot_labels()

    return plot.fig
