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


def makeFigure():
    """Must generate figure 4 first to populate cache."""
    alter = Alter()
    subject_class = alter.get_subject_class()

    Rtot = pd.read_csv(ALTER_RTOT_CACHE_PATH)
    Rtot.set_index(["Sample", "Antigen"], inplace=True)

    fucose_inferred = compute_fucose_ratio(Rtot).reset_index(level="Antigen")
    fucose_inferred.replace("gp120.Du156.12", "gp120.Du156", inplace=True)
    fucose_inferred.replace("IIIb.pr55.Gag", "pr55.Gag.IIIb", inplace=True)
    df_compare = pd.merge(
        fucose_inferred, subject_class, how="inner", on="Sample"
    ).reset_index()

    plot = Multiplot(
        (10, 12),
        # the plotting for this function breaks at our required subplot
        # dimensions, so we make the figure larger than it needs to be
        fig_size=(7.5, 7.5),
        subplot_specs=[
            (0, 10, 0, 3),
            (0, 10, 3, 3),
            (0, 3, 6, 6),
            (4, 5, 6, 6),
        ],
    )

    # a
    ax = plot.axes[0]
    sns.boxplot(
        data=df_compare,
        x="Antigen",
        y="fucose_inferred",
        hue="class",
        ax=ax,
        hue_order=["EC", "VC", "TP", "UP"],
        showfliers=False,
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
        x="Antigen Category",
        y="fucose_inferred",
        ax=ax,
        showfliers=False,
    )
    ax.set_ylabel("IgG Fucosylation (%)")
    ax.set_ylim(*Y_LIM)
    ax.set_xlabel("Antigen type")
    ax.set_xticklabels(ax.get_xticklabels())
    pairs = [("Env trimer", "p24"), ("Env trimer", "pr55.Gag")]
    annotator = Annotator(
        ax, pairs, data=df_compare, x="Antigen Category", y="fucose_inferred"
    )
    annotate_mann_whitney(annotator)

    ax = plot.axes[3]
    fucose_inferred = fucose_inferred.reset_index().set_index("Antigen")
    fucose_inferred = fucose_inferred.pivot_table(
        index="Antigen", columns="Sample", values="fucose_inferred"
    )
    corr = fucose_inferred.T.corr()
    sns.heatmap(
        data=corr,
        yticklabels=True,
        xticklabels=True,
        center=0,
        ax=ax,
        cbar_kws={"label": "Pearson correlation"},
    )
    # remove the colorbar when generating figures for layout and add it manually
    # cbar = ax.collections[0].colorbar
    # cbar.remove()

    plot.add_subplot_labels(ax_relative=True)
    # plot.fig.tight_layout(pad=0, w_pad=0, h_pad=-1)

    return plot.fig
