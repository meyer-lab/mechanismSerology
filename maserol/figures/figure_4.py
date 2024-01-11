from pathlib import Path

import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from tensordata.alter import load_file, data as alter

from maserol.preprocess import Rtot_to_xarray
from maserol.core import optimize_loss
from maserol.preprocess import assemble_options, prepare_data
from maserol.figures.common import getSetup, add_subplot_labels


def makeFigure():
    axes, fig = getSetup((8, 3), (1, 2))
    figure_4b(axes[1])
    add_subplot_labels(axes)
    return fig


def figure_4b(ax):
    # load data
    data = alter()["Fc"]
    data = data.sel(
        Receptor=[
            "FcgRIIa.H131",
            "FcgRIIb",
            "FcgRIIIa.V158",
            "FcgRIIIb",
            "IgG1",
            "IgG3",
        ]
    )
    translate = {
        "FcgRIIa.H131": "FcgRIIA-131H",
        "FcgRIIa.R131": "FcgRIIA-131R",
        "FcgRIIIa.F158": "FcgRIIIA-158F",
        "FcgRIIIa.V158": "FcgRIIIA-158V",
    }
    data = data.assign_coords(
        Receptor=[translate.get(r, r) for r in data.Receptor.values]
    )
    data = prepare_data(data)
    data = data.sel(
        Complex=[
            complex for complex in data.Complex.values if complex[1] == "gp120.SF162"
        ]
    )
    glycans = load_file("data-glycan-gp120")
    glycans = glycans.rename(columns={"subject": "Sample"})
    glycans = glycans[
        glycans["F.total"] > 1
    ]  # there is one 0 sample that is bad (other measurements for that sample are 0 too)

    rcps = ["IgG1", "IgG1f", "IgG3", "IgG3f"]
    opts = assemble_options(data, rcps=rcps)
    params, _ = optimize_loss(data, **opts, return_reshaped_params=True)
    Rtot = Rtot_to_xarray(params["Rtot"], data, rcps=rcps)
    Rtot_SF162_df = (
        Rtot.to_dataframe(name="Abundance")
        .drop(columns=["Sample", "Antigen"])
        .reset_index()
        .pivot(index="Sample", columns="Receptor", values="Abundance")
        .reset_index()
    )
    Rtot_SF162_df.columns.name = None
    df_comb = pd.merge(left=glycans, right=Rtot_SF162_df, on="Sample", how="inner")
    df_comb["F.total Inferred"] = (
        (df_comb["IgG1f"] + df_comb["IgG3f"])
        / (df_comb["IgG1"] + df_comb["IgG1f"] + df_comb["IgG3"] + df_comb["IgG3f"])
        * 100
    )
    sns.scatterplot(data=df_comb, x="F.total", y="F.total Inferred", ax=ax)
    ax.set_xlabel("Measured IgG Fucosylation (%)")
    ax.set_ylabel("Inferred IgG Fucosylation (%)")
    r, p = pearsonr(df_comb["F.total"], df_comb["F.total Inferred"])
    ax.set_title(
        "Model Inferences vs CE Measurements of IgG Fucosylation"
    )
    ax.text(
        0.8,
        0.05,
        r"r=" + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.text(
        0.8,
        0.01,
        r"p=" + "{:.2e}".format(p),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
