""" Import binding affinities. """

from os.path import join, dirname
import pandas as pd
import xarray as xr

path_here = dirname(dirname(__file__))


def human_affinity():
    # TODO: change this into xarray format

    return pd.read_csv(join(path_here, "maserol/data/murine-affinities.csv"),
                       delimiter=",", comment="#", index_col=0)


def assemble_Kav(data: xr.DataArray):
    # TODO: omit entries irrelavant to IgG or FcRg, then put the matrix together with labels
    pass