import numpy as np
import pandas as pd
import seaborn as sns
from tensordata.alter import load_file, data as alter

from maserol.core import Rtot_to_xarray, optimize_loss, reshape_params
from maserol.preprocess import assemble_options, prepare_data
from maserol.figures.common import getSetup


def makeFigure():
    axes, fig = getSetup((4, 3), (1, 1))

    # load data
    data = alter()["Fc"]
    data = data.sel(
        Receptor=[
            "FcgRIIa.H131",
            "FcgRIIa.R131",
            "FcgRIIb",
            "FcgRIIIa.F158",
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

    # infer fucose
    rcps = ["IgG1", "IgG1f", "IgG3", "IgG3f"]
    opts = assemble_options(data, rcps=rcps)
    params, _ = optimize_loss(data, **opts, return_reshaped_params=True)
    Rtot = Rtot_to_xarray(params["Rtot"], data, rcps=rcps)
    Rtot_SF162 = Rtot.sel(
        Complex=[cplx for cplx in Rtot.Complex.values if cplx[1] == "gp120.SF162"]
    )
    Rtot_SF162_df = (
        Rtot_SF162.to_dataframe(name="Abundance")
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
    # filter out samples which have MS measurements less than 60
    df_comb_filtered = df_comb[df_comb["F.total"] > 60]
    sns.scatterplot(
        data=df_comb_filtered, x="F.total", y="F.total Inferred", ax=axes[0]
    )
    r = np.corrcoef(df_comb_filtered["F.total"], df_comb_filtered["F.total Inferred"])[
        0, 1
    ]
    axes[0].set_title(
        "MS Fucose Ratio vs Inferred Fucose Ratio (filtered by MS fucose ratio > 60)"
    )
    axes[0].text(
        0.95,
        0.05,
        r"r=" + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=axes[0].transAxes,
    )
    return fig
