from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from maserol.core import infer_Lbound
from maserol.figures.common import getSetup
from maserol.preprocess import assemble_Ka, assemble_options


def lbound_cube_to_df(cube: xr.DataArray) -> pd.DataFrame:
    """
    Convert lbound as an xarray into a dataframe, where columns are named as <receptor>_<antigen>.

    Args:
      cube: xr.DataArray for lbound (detection reagent) signals. Should have
        dimensions Sample x Receptor x Antigen.

    Returns:
      pd.DataFrame with lbound data.
    """

    df = cube.to_dataframe(name="abundance").reset_index()
    df["rcp_antigen"] = df["Receptor"].astype(str) + "_" + df["Antigen"].astype(str)
    return (
        df.pivot(values="abundance", index="Sample", columns="rcp_antigen")
        .reset_index()
        .rename_axis(None, axis=1)
        .copy()
    )


def rcp_cube_to_df(cube: xr.DataArray) -> pd.DataFrame:
    """
    Convert Rtot as an xarray into a dataframe, where columns are named as AB_<receptor>_<antigen>.

    Args:
      cube: xr.DataArray for Rtot (Ab abundances). Should have
        dimensions Sample x Antibody x Antigen.

    Returns:
      pd.DataFrame with Rtot data.
    """
    df = cube.to_dataframe(name="abundance").reset_index()
    df["ab_antigen"] = (
        "AB_" + df["Antibody"] + "_" + df["Antigen"]
    )  # add "AB_" prefix to prevent clashes with lbound cube
    return (
        df.pivot_table(values="abundance", index="Sample", columns="ab_antigen")
        .reset_index()
        .rename_axis(None, axis=1)
    )


def construct_lbound_rcp_df(cube: xr.DataArray, Rtot: xr.DataArray, **opt_opts):
    """
    Construct a dataframe with true lbound, inferred lbound, and Ab abundances.
    Column names:
      true lbound: <receptor>_<antigen>
      inferred lbound: <receptor>_<antigen>_inferred
      rtot (ab abundances): AB_<receptor>_<antigen>

    Args:
      cube: xr.DataArray for lbound (detection reagent signals). Should have
        dimensions Sample x Receptor x Antigen.
      Rtot: xr.DataArray for rtot (ab abundances) after fitting. Should have
        dimensions Sample x Antibody x Antigen.

    Returns:
      pd.DataFrame with Rtot data.
    """
    ab_types = Rtot.Antibody.values
    Ka = assemble_Ka(cube, ab_types=ab_types)
    opts = opt_opts or assemble_options(cube, ab_types)
    Lbound_nd = infer_Lbound(
        Rtot.values, Ka.values, opts["L0"], opts["KxStar"], opts["f"]
    )
    Lbound = cube.copy()
    Lbound.values = Lbound_nd
    df_detection = lbound_cube_to_df(cube)
    df_ab = rcp_cube_to_df(Rtot)
    df_inferred = lbound_cube_to_df(Lbound)
    rename_dict = {}
    for col in df_inferred.columns:
        if col == "Sample":
            continue
        rename_dict[col] = col + "_inferred"
    df_inferred = df_inferred.rename(columns=rename_dict)
    df = pd.merge(
        left=pd.merge(left=df_detection, right=df_inferred, on="Sample"),
        right=df_ab,
        on="Sample",
    )
    return df


def plot_lbound(
    samples: List,
    antigen: str,
    cube: xr.DataArray,
    Rtot: xr.DataArray,
    ax=None,
    sample_labels: List[str] = None,
):
    """
    Plots actual lbound vs inferred lbound as bar graph.

    Args:
      samples: List of samples (along Sample dimension in cube) to plot.
      antigen: Single antigen to plot for each sample.
      cube: Actual lbound
      Rtot: Ab abundances
      ax: axis to plot on
      sample_labels: same length as samples. How to show each sample on the
        legend. If not passed, the legend will show 'Sample <i>'
      FcIdx: index at which Fc detection reagents appear in cube.

    Returns:
      matplotlib axis on which plot is shown
    """
    df = construct_lbound_rcp_df(cube, Rtot)

    if ax is None:
        _, ax = plt.subplots()

    df["Sample"] = pd.Categorical(df["Sample"], categories=samples, ordered=True)
    df = df.sort_values("Sample")
    df_sub = df[df["Sample"].isin(samples)]
    assert np.all(df_sub["Sample"].values == np.array(samples))
    cols1 = [f"{rcp}_{antigen}" for rcp in cube.Receptor.values]
    cols2 = [f"{col}_inferred" for col in cols1]

    interleaved = [val for pair in zip(cols1, cols2) for val in pair]
    dfr = df_sub[interleaved]
    dfr = np.log10(dfr)
    values = np.nan_to_num(dfr.values, neginf=0)
    dfr = pd.DataFrame(values, columns=dfr.columns)

    # Create x-axis values with space after every 2 xticks
    x_values = np.arange(len(interleaved)) + np.repeat(
        np.arange(len(interleaved) // 2), 2
    )

    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(len(samples))]
    for i in range(len(samples)):
        ax.bar(x_values + i * 0.2, dfr.iloc[i], width=0.2, label=sample_labels[i])
    ax.legend()
    ax.set_xticks(x_values)
    ax.set_xticklabels(interleaved, rotation=90)
    ax.set_ylabel("Log abundance")
    ax.set_title("Detection Reagent Abundances")
    return ax


def plot_rcp(
    samples: List,
    antigen: str,
    Rtot: xr.DataArray,
    ax: Optional[plt.axis] = None,
    sample_labels: Optional[List[str]] = None,
):
    """
    Plots ab abundances as bar plot.

    Args:
      samples: List of samples (along Sample dimension in cube) to plot.
      antigen: Single antigen to plot for each sample.
      Rtot: Ab abundances
      ax: axis to plot on
      sample_labels: same length as samples. How to show each sample on the
        legend. If not passed, the legend will show 'Sample <i>'

    Returns:
      matplotlib axis on which plot is shown
    """
    df = rcp_cube_to_df(Rtot)
    ab_types = Rtot.Antibody.values
    if ax is None:
        _, ax = plt.subplots()
    df["Sample"] = pd.Categorical(df["Sample"], categories=samples, ordered=True)
    df = df.sort_values("Sample")
    df_sub = df[df["Sample"].isin(samples)]
    assert np.all(df_sub["Sample"].values == np.array(samples))
    dfa = df_sub[[f"AB_{ab}_{antigen}" for ab in ab_types]]
    dfa = np.log10(dfa)
    dfa.transpose().plot.bar(ax=ax)
    ax.legend(sample_labels or [f"Sample {i}" for i in range(1, len(samples) + 1)])
    ax.set_xticks(np.arange(Rtot.sizes["Antibody"]), ab_types)
    ax.set_ylabel("Log abundance")
    ax.set_title("Fitted Ab Abundances")


def plot_sample_fit(
    samples: List,
    antigen: str,
    cube: xr.DataArray,
    Rtot: xr.DataArray,
    sample_labels=None,
):
    """
    Shows plot_lbound and plot_rcp in adjacent plots. See those functions for
    details.

    Args:
      samples: List of samples (along Sample dimension in cube) to plot.
      antigen: Single antigen to plot for each sample.
      cube: Actual lbound
      Rtot: Ab abundances
      sample_labels: same length as samples. How to show each sample on the
        legend. If not passed, the legend will show 'Sample <i>'
      FcIdx: index at which Fc detection reagents appear in cube.

    Returns:
      matplotlib figure on which plots are shown
    """
    axes, _ = getSetup((15, 7), (1, 2))
    plot_lbound(samples, antigen, cube, Rtot, ax=axes[0], sample_labels=sample_labels)
    plot_rcp(samples, antigen, Rtot, ax=axes[1], sample_labels=sample_labels)
