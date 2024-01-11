from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensordata
from statannotations.Annotator import Annotator
from tensordata.kaplonekVaccineSA import data as get_covid_vaccination_data
from tensordata.zohar import data as zohar

from maserol.core import optimize_loss
from maserol.figures.common import getSetup, add_subplot_label
from maserol.preprocess import prepare_data, assemble_options, Rtot_to_df

THIS_DIR = Path(__file__).parent
CACHE_DIR = THIS_DIR.parent / "data" / "cache"
UPDATE_CACHE = False


def makeFigure():
    axes, fig = getSetup((10, 5), (2, 3), multz={3: 2})
    figure_5a(axes[0])
    figure_5b(axes[3])
    add_subplot_label(axes[0], 'a')
    add_subplot_label(axes[3], 'b')
    return fig


def figure_5a(ax):
    data = prepare_data(zohar())
    data = data.sel(Ligand=[l for l in data.Ligand.values if l != "IgG2"])
    raw_data = pd.read_csv(tensordata.zohar.DATA_PATH)
    raw_data.rename(columns={"sample_ID": "Sample"}, inplace=True)
    opts = assemble_options(data)

    if UPDATE_CACHE:
        x, ctx = optimize_loss(data, **opts, return_reshaped_params=True)
        df = Rtot_to_df(x["Rtot"], data, rcps=list(opts["rcps"]))
        df = df.reset_index()
        df.to_csv(CACHE_DIR / "fig_5a_Rtot.csv")
    else:
        df = pd.read_csv(CACHE_DIR / "fig_5a_Rtot.csv")

    df = df[df["Antigen"] == "S"]

    df["Fucose Ratio"] = (
        (df["IgG1f"] + df["IgG3f"])
        / (df["IgG1"] + df["IgG1f"] + df["IgG3"] + df["IgG3f"])
        * 100
    )

    df_merged = pd.merge(df, raw_data, how="inner", on="Sample")
    df_merged.sort_values("ARDS", inplace=True)

    sns.boxplot(data=df_merged, x="ARDS", y="Fucose Ratio", ax=ax)
    ax.set_xlabel(None)
    ax.set_xticklabels(["Non-ARDS", "ARDS"])
    ax.set_ylabel("anti-S IgG Fucosylation (%)")

    pairs = (("Yes", "No"),)
    annotator = Annotator(ax, pairs, data=df_merged, x="ARDS", y="Fucose Ratio")
    annotator.configure(test="t-test_ind", text_format="star")
    annotator.apply_and_annotate()


def figure_5b(ax):
    covid_vaccination_data = get_covid_vaccination_data()

    # prepare luminex
    lum_data = prepare_data(covid_vaccination_data["Luminex"])

    # prepare meta
    meta_data = (
        covid_vaccination_data["Meta"]
        .to_dataframe()
        .reset_index(level="Metadata")
        .pivot(columns="Metadata", values="Meta")
    )
    meta_data.columns.name = None
    meta_data.index.name = "Sample"

    if UPDATE_CACHE:
        opts = assemble_options(lum_data)
        params, _ = optimize_loss(lum_data, **opts, return_reshaped_params=True)
        df = Rtot_to_df(params["Rtot"], lum_data, list(opts["rcps"])).reset_index(
            level="Antigen"
        )
        df.to_csv(CACHE_DIR / "fig_5b_Rtot.csv")
    else:
        df = pd.read_csv(CACHE_DIR / "fig_5b_Rtot.csv")
        df.set_index("Sample", inplace=True)

    df["fucose"] = (
        (df["IgG1f"] + df["IgG3f"])
        / (df["IgG1"] + df["IgG1f"] + df["IgG3"] + df["IgG3f"])
        * 100
    )
    df = df.merge(meta_data, on="Sample", how="inner")

    fig = plt.figure(figsize=(13, 5))
    ag_exclude = [
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
    df_sub = df[~df["Antigen"].isin(ag_exclude)]
    df_sub = df_sub.sort_values("Antigen")[::-1]

    sns.boxplot(
        data=df_sub,
        x="Antigen",
        y="fucose",
        hue="infection.status",
        ax=ax,
        hue_order=["control", "case"],
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel("IgG Fucosylation (%)")
    ax.set_xlabel("Antigen")
    pairs = [((ag, "control"), (ag, "case")) for ag in df_sub.Antigen.unique()]
    annotator = Annotator(
        ax, pairs, data=df_sub, x="Antigen", y="fucose", hue="infection.status"
    )
    annotator.configure(test="t-test_ind", text_format="star")
    annotator.apply_and_annotate()

    ax.legend(title=None)
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["Healthy", "COVID-19$^+$"]
    ax.legend(handles, new_labels)

    sns.move_legend(ax, "lower right")

    return fig
