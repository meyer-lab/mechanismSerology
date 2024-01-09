import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensordata
from statannotations.Annotator import Annotator
from tensordata.zohar import data as zohar

from maserol.preprocess import prepare_data, assemble_options, Rtot_to_df
from maserol.core import optimize_loss


def makeFigure():
    data = prepare_data(zohar())
    data = data.sel(Ligand=[l for l in data.Ligand.values if l != "IgG2"])
    raw_data = pd.read_csv(tensordata.zohar.DATA_PATH)
    raw_data.rename(columns={"sample_ID": "Sample"}, inplace=True)
    opts = assemble_options(data)
    x, ctx = optimize_loss(data, **opts, return_reshaped_params=True)
    df = Rtot_to_df(x["Rtot"], data, rcps=list(opts["rcps"]))
    df = df.reset_index()
    df = df[df["Antigen"] == "S"]

    df["Fucose Ratio"] = (df["IgG1f"] + df["IgG3f"]) / (
        df["IgG1"] + df["IgG1f"] + df["IgG3"] + df["IgG3f"]
    )

    df_merged = pd.merge(df, raw_data, how="inner", on="Sample")
    df_merged.sort_values("ARDS", inplace=True)

    fig = plt.figure(figsize=(5, 4))
    ax = sns.boxplot(data=df_merged, x="ARDS", y="Fucose Ratio")
    ax.set_xlabel(None)
    ax.set_xticklabels(["Non-ARDS", "ARDS"])
    ax.set_ylabel("Fucosylated anti-S IgG (%)")

    pairs = (("Yes", "No"),)
    annotator = Annotator(ax, pairs, data=df_merged, x="ARDS", y="Fucose Ratio")
    annotator.configure(test="t-test_ind", text_format="star")
    annotator.apply_and_annotate()
    return fig
