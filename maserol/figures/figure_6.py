import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

from maserol.datasets import Alter
from maserol.figures.common import Multiplot, annotate_mann_whitney
from maserol.figures.figure_4 import ALTER_RTOT_CACHE_PATH
from maserol.util import compute_fucose_ratio

Y_LIM = (-2, 102)
X_LABEL_ROTATION = 40

TOP_PLOTS = True


def makeFigure():
    """Must generate figure 4 first to populate cache."""
    alter = Alter()
    subject_class = alter.get_subject_class()

    Rtot = pd.read_csv(ALTER_RTOT_CACHE_PATH)
    Rtot.set_index(["Sample", "Antigen"], inplace=True)

    palette = sns.color_palette("tab10").as_hex()

    fucose_inferred = compute_fucose_ratio(Rtot).reset_index(level="Antigen")
    fucose_inferred.replace("gp120.Du156.12", "gp120.Du156", inplace=True)
    fucose_inferred.replace("IIIb.pr55.Gag", "pr55.Gag.IIIb", inplace=True)
    df_compare = pd.merge(
        fucose_inferred, subject_class, how="inner", on="Sample"
    ).reset_index()

    plot = Multiplot(
        (11, 13) if TOP_PLOTS else (10, 12),
        # the plotting for this function breaks at our required subplot
        # dimensions, so we make the figure larger than it needs to be
        fig_size=(8, 7.5) if TOP_PLOTS else (9, 9),
        subplot_specs=[
            (0, 10, 0, 3),
            (0, 10, 3, 3),
            (0, 4, 6, 3) if TOP_PLOTS else (0, 3, 6, 3),
            (0, 4, 9, 5) if TOP_PLOTS else (0, 3, 9, 4),
            (6, 6, 6, 6) if TOP_PLOTS else (5, 7, 6, 6),
        ],
    )

    # pr55.Gag first, then gps, then SOSIP, then p24
    ag_order = (
        [elem for elem in df_compare.Antigen.unique() if "pr55" in elem]
        + [elem for elem in df_compare.Antigen.unique() if "gp" in elem]
        + [elem for elem in df_compare.Antigen.unique() if "SOSIP" in elem]
        + [elem for elem in df_compare.Antigen.unique() if "p24" in elem]
    )

    assert len(ag_order) == len(df_compare.Antigen.unique())

    # a
    ax = plot.axes[0]
    sns.boxplot(
        data=df_compare,
        x="Antigen",
        y="fucose_inferred",
        hue="class",
        ax=ax,
        hue_order=["EC", "VC", "TP", "UP"],
        order=ag_order,
        showfliers=False,
        palette=[palette[2], palette[8], palette[1], palette[3]],
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=X_LABEL_ROTATION)
    ax.set_xlabel("Antigen", labelpad=0)
    ax.set_ylabel("IgG Fucosylation (%)")
    ax.set_ylim(*Y_LIM)
    ax.legend(title=None)
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        "Elite controller",
        "Viremic controller",
        "Treated progressor",
        "Untreated progressor",
    ]
    legend = ax.legend(handles, new_labels, loc="lower right")
    legend.get_frame().set_alpha(1)

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
        showfliers=False,
        order=ag_order,
        palette=[palette[2], palette[7]],
    )
    ax.set_xlabel("Antigen", labelpad=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=X_LABEL_ROTATION)
    ax.set_ylabel("IgG Fucosylation (%)")
    ax.set_ylim(*Y_LIM)
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["Elite controller", "Others"]
    ax.legend(handles, new_labels)
    sns.move_legend(ax, "lower right")
    pairs = [((ag, False), (ag, True)) for ag in df_compare.Antigen.unique()]
    annotator = Annotator(
        ax, pairs, data=df_compare, x="Antigen", y="fucose_inferred", hue="is_EC"
    )
    annotate_mann_whitney(annotator, correction=None)

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
            df_compare.loc[i, "Antigen Category"] = "pr55.Gag"
    sns.boxplot(
        data=df_compare,
        y="Antigen Category",
        x="fucose_inferred",
        orient="h",
        ax=ax,
        showfliers=False,
        palette=palette[0:1],
    )
    ax.set_xlabel("IgG Fucosylation (%)")
    ax.set_xlim(*Y_LIM)
    ax.set_ylabel("Antigen type")
    pairs = [("Env trimer", "pr55.Gag")]
    annotator = Annotator(
        ax,
        pairs,
        data=df_compare,
        y="Antigen Category",
        x="fucose_inferred",
        orient="h",
    )
    annotate_mann_whitney(annotator)

    ax = plot.axes[3]
    sns.boxplot(
        data=df_compare,
        y="Antigen Category",
        x="fucose_inferred",
        hue="is_EC",
        orient="h",
        ax=ax,
        showfliers=False,
        hue_order=[True, False],
        palette=[palette[2], palette[7]],
    )
    ax.set_xlabel("IgG Fucosylation (%)")
    ax.set_xlim(*Y_LIM)
    ax.set_ylabel("Antigen type")
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["Elite controller", "Others"]
    ax.legend(handles, new_labels)
    pairs = [
        ((ag, True), (ag, False)) for ag in df_compare["Antigen Category"].unique()
    ]
    annotator = Annotator(
        ax,
        pairs,
        data=df_compare,
        y="Antigen Category",
        x="fucose_inferred",
        hue="is_EC",
        orient="h",
    )
    annotate_mann_whitney(annotator)

    ax = plot.axes[4]
    fucose_inferred = fucose_inferred.reset_index().set_index("Antigen")
    fucose_inferred = fucose_inferred.pivot_table(
        index="Antigen", columns="Sample", values="fucose_inferred"
    )
    corr = fucose_inferred.T.corr()
    corr = corr.loc[ag_order, ag_order]
    sns.heatmap(
        data=corr,
        yticklabels=True,
        xticklabels=True,
        center=0,
        ax=ax,
        cbar_kws={"label": "Pearson correlation"},
    )
    # remove the colorbar when generating figures for layout and add it manually
    cbar = ax.collections[0].colorbar
    cbar.remove()

    plot.add_subplot_labels(ax_relative=True)

    return plot.fig
