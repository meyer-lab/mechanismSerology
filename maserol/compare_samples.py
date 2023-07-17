from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from maserol.core import DEFAULT_FC_IDX_VAL, inferLbound
from maserol.figures.common import getSetup
from maserol.preprocess import assembleKav


def lbound_cube_to_df(cube: xr.DataArray):
    df = cube.to_dataframe(name="abundance").reset_index()
    df["rcp_antigen"] = df["Receptor"] + "_" + df["Antigen"]
    return df.pivot_table(values='abundance', index='Sample', columns='rcp_antigen').reset_index().rename_axis(None, axis=1)

def rcp_cube_to_df(cube: xr.DataArray):
    df = cube.to_dataframe(name="abundance").reset_index()
    df["ab_antigen"] = "AB_" + df["Antibody"] + "_" + df["Antigen"] # add "AB_" prefix to prevent clashes with lbound cube
    return df.pivot_table(values='abundance', index='Sample', columns='ab_antigen').reset_index().rename_axis(None, axis=1)

def construct_lbound_rcp_df(cube: xr.DataArray, Rtot: xr.DataArray, FcIdx=DEFAULT_FC_IDX_VAL):
    ab_types = Rtot.Antibody.values
    Ka = assembleKav(cube, ab_types=ab_types)
    Lbound_nd = inferLbound(cube.values, Rtot.values, Ka.values, FcIdx=FcIdx)
    Lbound = cube.copy()
    Lbound.values = Lbound_nd
    df_detection = lbound_cube_to_df(cube)
    df_ab = rcp_cube_to_df(Rtot)
    df_inferred = lbound_cube_to_df(Lbound)
    rename_dict = {}
    for col in df_inferred.columns:
        if col == "Sample": continue
        rename_dict[col] = col + "_inferred"
    df_inferred = df_inferred.rename(columns=rename_dict)
    df = pd.merge(left=pd.merge(left=df_detection, right=df_inferred, on="Sample"), right=df_ab, on="Sample")
    return df

def plot_lbound(samples: List, antigen: str, cube: xr.DataArray, Rtot: xr.DataArray, ax=None, sample_labels: List[str]=None, FcIdx=DEFAULT_FC_IDX_VAL):
    df = construct_lbound_rcp_df(cube, Rtot, FcIdx=FcIdx)

    if ax is None:
        _, ax = plt.subplots()

    df_sub = df[df["Sample"].isin(samples)]
    cols1 = [f"{rcp}_{antigen}" for rcp in cube.Receptor.values]
    cols2 = [f"{col}_inferred" for col in cols1]

    interleaved = [val for pair in zip(cols1, cols2) for val in pair]
    dfr = df_sub[interleaved]
    dfr = np.log10(dfr)
    dfr = pd.DataFrame(np.nan_to_num(dfr.values), columns=dfr.columns)

    # Create x-axis values with space after every 2 xticks
    x_values = np.arange(len(interleaved)) + np.repeat(np.arange(len(interleaved)//2), 2)

    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(len(samples))]
    for i in range(len(samples)):
        ax.bar(x_values + i*0.2, dfr.iloc[i], width=0.2, label=sample_labels[i])
    ax.legend()
    ax.set_xticks(x_values)
    ax.set_xticklabels(interleaved, rotation=90) 
    ax.set_ylabel("Log abundance")
    ax.set_title("Detection Reagent Abundances")

def plot_rcp(samples: List, antigen: str, Rtot: xr.DataArray, ax=None, sample_labels=None):
    df = rcp_cube_to_df(Rtot)
    ab_types = Rtot.Antibody.values
    if ax is None:
        _, ax = plt.subplots()
    dfa = df[df["Sample"].isin(samples)][[f"AB_{ab}_{antigen}" for ab in ab_types]]
    dfa = np.log10(dfa)
    dfa.transpose().plot.bar(ax=ax)
    ax.legend(sample_labels or [f"Sample {i}" for i in range(1, len(samples) + 1)])
    ax.set_xticks(np.arange(Rtot.sizes["Antibody"]), ab_types)
    ax.set_ylabel("Log abundance")
    ax.set_title("Fitted Ab Abundances")

def plot_sample_fit(samples: List, antigen: str, cube: xr.DataArray, Rtot: xr.DataArray, sample_labels=None, FcIdx=DEFAULT_FC_IDX_VAL):
    axes, _ = getSetup((15, 7), (1, 2))
    plot_lbound(samples, antigen, cube, Rtot, ax=axes[0], sample_labels=sample_labels, FcIdx=FcIdx)
    plot_rcp(samples, antigen, Rtot, ax=axes[1], sample_labels=sample_labels)
