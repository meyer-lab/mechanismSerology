import pandas as pd
import seaborn as sns
from itertools import combinations
from tensordata.alter import load_file, data as alter

from maserol.figures.common import getSetup

translate = {
    "FcgRIIa.H131": "FcgRIIA-131H",
    "FcgRIIa.R131": "FcgRIIA-131R",
    "FcgRIIIa.F158": "FcgRIIIA-158F",
    "FcgRIIIa.V158": "FcgRIIIA-158V",
}


def makeFigure():
    axes, fig = getSetup((9, 4), (1, 2))

    data = alter()["Fc"]
    data = data.assign_coords(
        Receptor=[translate.get(r, r) for r in data.Receptor.values]
    )
    df_lum = data.to_dataframe(name="signal").reset_index()
    df_lum["rcp_antigen"] = df_lum["Receptor"] + "_" + df_lum["Antigen"]

    df_lum = (
        df_lum.pivot_table(values="signal", index="Sample", columns="rcp_antigen")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    glycans = load_file("data-glycan-gp120")
    glycans = glycans.rename(columns={"subject": "Sample"})

    df = pd.merge(glycans, df_lum, how="inner", on="Sample")

    plot_diffs(df, axes[0])
    axes[0].set_title(
        "Pairwise differences in receptor amounts vs pairwise difference in MS fucose ratio"
    )
    plot_diffs(df[df["F.total"] > 60], axes[1])
    axes[1].set_title(
        "Pairwise differences in receptor amounts vs pairwise difference in MS fucose ratio (F.total > 60)"
    )
    return fig


def plot_diffs(df, ax):
    # Let's assume df is your dataframe
    pairs = list(combinations(df.index, 2))

    # Filter pairs to ensure 'F.total' of the first is always greater than the second
    pairs = [
        (a, b) if df.loc[a, "F.total"] > df.loc[b, "F.total"] else (b, a)
        for a, b in pairs
    ]

    # Compute pairwise differences
    diff_df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(pairs, names=["first", "second"])
    )

    diff_df["FcgRIIIA_diff"] = (
        df.loc[
            diff_df.index.get_level_values("first"), "FcgRIIA-131H_gp120.SF162"
        ].values
        - df.loc[
            diff_df.index.get_level_values("second"), "FcgRIIA-131H_gp120.SF162"
        ].values
    )
    diff_df["FcgRIIA_diff"] = (
        df.loc[
            diff_df.index.get_level_values("first"), "FcgRIIIA-158V_gp120.SF162"
        ].values
        - df.loc[
            diff_df.index.get_level_values("second"), "FcgRIIIA-158V_gp120.SF162"
        ].values
    )
    diff_df["F_total_diff"] = (
        df.loc[diff_df.index.get_level_values("first"), "F.total"].values
        - df.loc[diff_df.index.get_level_values("second"), "F.total"].values
    )

    # Divide pairwise difference of FcgRIIIA by the pairwise difference of FcgRIIA
    diff_df["FcgRIIIA/FcgRIIA"] = diff_df["FcgRIIIA_diff"] / diff_df["FcgRIIA_diff"]

    sns.scatterplot(
        data=diff_df,
        x="FcgRIIIA_diff",
        y="FcgRIIA_diff",
        hue="F_total_diff",
        alpha=0.6,
        s=5,
        ax=ax,
    )
