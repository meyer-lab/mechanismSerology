from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from statannotations.Annotator import Annotator

from tensordata.alter import data as alter, load_file
from maserol.core import optimize_loss
from maserol.figures.common import getSetup, add_subplot_labels
from maserol.preprocess import prepare_data, assemble_options, Rtot_to_df

THIS_DIR = Path(__file__).parent
CACHE_DIR = THIS_DIR.parent / "data" / "cache"
UPDATE_CACHE = False


def makeFigure():
    data = alter()["Fc"]
    data = data.sel(
        Receptor=[
            "IgG1",
            "IgG3",
            "FcgRIIa.H131",
            "FcgRIIb",
            "FcgRIIIa.V158",
            "FcgRIIIb",
        ]
    )
    translate = {
        "FcgRIIa.H131": "FcgRIIA-131H",
        "FcgRIIIa.V158": "FcgRIIIA-158V",
    }
    data = data.assign_coords(
        Receptor=[translate.get(r, r) for r in data.Receptor.values]
    )
    data = prepare_data(data)
    subject_class = (
        load_file("meta-subjects")
        .rename(columns={"subject": "Sample"})
        .set_index("Sample")
    )

    rcps = ["IgG1", "IgG1f", "IgG3", "IgG3f"]
    opts = assemble_options(data, rcps=rcps)
    if UPDATE_CACHE:
        params, _ = optimize_loss(data, **opts, return_reshaped_params=True)
        df = Rtot_to_df(params["Rtot"], data, rcps)
        df.to_csv(CACHE_DIR / "fig_6_Rtot.csv")
    else:
        df = pd.read_csv(CACHE_DIR / "fig_6_Rtot.csv")
        df.set_index(["Sample", "Antigen"], inplace=True)

    get_fucose = (
        lambda df: (df["IgG1f"] + df["IgG3f"])
        / (df["IgG1"] + df["IgG1f"] + df["IgG3"] + df["IgG3f"])
        * 100
    )
    df["f"] = get_fucose(df)
    df["class"] = subject_class.loc[df.index.get_level_values("Sample")][
        "class.etuv"
    ].values

    assert (
        df.loc[(100681, "gp120.Du156.12"), "class"]
        == subject_class.loc[100681, "class.etuv"]
    )

    df = df.reset_index()

    axes, fig = getSetup((12, 11), (3, 2), multz={0: 1, 2: 1})
    y = "f"
    hue = "class"
    ax = axes[0]
    sns.boxplot(data=df, x="Antigen", y=y, hue=hue, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontsize="small")
    ax.set_ylabel("IgG Fucosylation (%)")
    ax.legend(title=None)
    sns.move_legend(ax, "lower right")

    df["is_EC"] = df["class"] == "EC"
    y = "f"
    hue = "is_EC"
    ax = axes[1]
    sns.boxplot(data=df, x="Antigen", y=y, hue=hue, ax=ax, hue_order=[True, False])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontsize="small")
    ax.set_ylabel("IgG Fucosylation (%)")
    handles, labels = ax.get_legend_handles_labels()

    new_labels = ["EC", "Others"]
    ax.legend(handles, new_labels)

    sns.move_legend(ax, "lower right")

    pairs = []
    for ag in df.Antigen.unique():
        df_ag = df[(df["Antigen"] == ag) & (~df[y].isna())]
        if (
            ttest_ind(
                df_ag[df_ag[hue] == True][y], df_ag[df_ag[hue] == False][y]
            ).pvalue
            < 0.05
        ):
            pairs.append(((ag, False), (ag, True)))

    if pairs:
        annotator = Annotator(ax, pairs, data=df, x="Antigen", y=y, hue=hue)
        annotator.configure(test="t-test_ind", text_format="star")
        annotator.apply_and_annotate()

    ax = axes[2]
    df["Antigen Category"] = np.ones(len(df))
    for i in df.index:
        antigen = df.loc[i, "Antigen"]
        if (
            antigen.startswith("gp120")
            or antigen.startswith("gp140")
            or antigen.startswith("gp41")
            or antigen == "SOSIP"
        ):
            df.loc[i, "Antigen Category"] = "Env trimer"
        elif antigen.startswith("p24"):
            df.loc[i, "Antigen Category"] = "p24"
        else:
            df.loc[i, "Antigen Category"] = antigen

    sns.boxplot(data=df, x="Antigen Category", y="f", ax=ax)
    ax.set_ylabel("IgG Fucosylation (%)")

    pairs = [("Env trimer", "p24"), ("Env trimer", "IIIb.pr55.Gag")]
    annotator = Annotator(ax, pairs, data=df, x="Antigen Category", y="f")
    annotator.configure(test="t-test_ind", text_format="star", loc="outside")
    annotator.apply_and_annotate()

    add_subplot_labels(axes)

    return fig
