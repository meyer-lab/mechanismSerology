""" Import binding affinities. """

from ast import Pass
from os.path import join, dirname
import numpy as np
import pandas as pd
import xarray as xr
import re

path_here = dirname(dirname(__file__))
initial_affinity = 10**8
mode_order = ["Sample", "Receptor", "Antigen"]

def human_affinity():
    # read in as a DataFrame
    df = pd.read_csv(join(path_here, "maserol/data/human-affinities.csv"),
                       delimiter=",", comment="#", index_col=0)
    df.drop(["FcgRIIA-131R", "FcgRIIB-232T", "FcgRIIIA-158F"], inplace=True)
    return df

def standardize_dim_order(data, order_list):
    ' '' Transposes data to be in ("Sample", "Antigen", "Receptor") order '''
    return data.transpose(order_list[0], order_list[1], order_list[2])

def omit_unnecessary_receptors(data, abs="IgG"):
    ''' Omits all receptor data from the initial data that does not pertain to the specified antibody'''
    data_receptors = data.Receptor.values

    # edit to account for receptors other than IgG
    wanted_receptors = [x for x in data_receptors if x.startswith(abs)] + \
                       [x for x in data_receptors if (x.startswith("FcR") and x != "FcRalpha")]

    return data.sel(Receptor=wanted_receptors), wanted_receptors

def get_affinity(affinities_df, receptor, abs):
    ''' Given a receptor and antibody pair and dataFrame of known affinity values,
        returns the associatied human affinity value'''
    # figure out of receptor uses iii or 1,2,3 system
    x = re.search("3|2|1|i+", receptor, flags=re.IGNORECASE)
    match = x.group()
    num = 0
    if (match == '1' or match.lower() == 'i'):
        num = "I"
    elif (match == '2' or match.lower() == 'ii'):
        num = "II"
    else:
        num = "III"

    # search for receptor match in affinities dataArray
    for r in list(affinities_df.index):
        r_regex = "fc[gr]*" + num + receptor[x.end()::]
        if re.match(r_regex, r, flags=re.IGNORECASE):
            return affinities_df.at[r,abs]
    return 0

def prepare_data(data: xr.DataArray):
    data = omit_unnecessary_receptors(data)
    data = standardize_dim_order(data, mode_order)

def assemble_Kav(data: xr.DataArray):
    abs = ["IgG1", "IgG2", "IgG3", "IgG4"]
    
    # omit entries irrelevant to IgG and FcRg
    data = prepare_data(data)
    receptors = data.Receptor.values

    # get known affinities
    affinities = human_affinity()

    # assemble matrix
    data_placeholder = np.zeros((len(receptors), len(abs)))
    Kav = xr.DataArray(data_placeholder, coords=[receptors, abs], dims=["Receptor", "Abs"])
    
    # separate into list of fc and igg receptors
    fc = [x for x in receptors if (re.match("fc[gr]*", x, flags=re.IGNORECASE) and x != "FcRalpha")]
    igg = [x for x in receptors if (re.match("^igg", x, flags=re.IGNORECASE))]

    # fill in all IgG - IgG pair affinity values
    for ab in abs:
        for ig in igg:
            if (ab == ig):
                Kav.loc[dict(Receptor=ig, Abs=ab)] = initial_affinity

    # fill in remaining affinity values
    for ab in abs:
        for r in fc:
            affinity = get_affinity(affinities, r, ab)
            Kav.loc[dict(Receptor=r, Abs=ab)] = affinity
        
    return Kav
