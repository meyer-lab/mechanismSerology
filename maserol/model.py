""" Import binding affinities. """

from os.path import join, dirname
import numpy as np
import pandas as pd
import xarray as xr
import re
from .fixkav_opt_helpers import absf

path_here = dirname(dirname(__file__))
initial_affinity = 10**8
mode_order = ["Sample", "Receptor", "Antigen"]

def human_affinity():
    """
    Return a dataFrame of known affinity measurments
    """
    df = pd.read_csv(join(path_here, "maserol/data/human-affinities.csv"),
                       delimiter=",", comment="#", index_col=0)
    df.drop(["FcgRIIA-131R", "FcgRIIB-232T", "FcgRIIIA-158F"], inplace=True)
    return df

def get_affinity(affinities_df, receptor, abs):
    """ 
    Given a receptor and antibody pair and dataFrame of known affinity values,
    returns the associatied human affinity value 
    """
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

def prepare_data(data: xr.DataArray, abs="IgG"):
    """
    Transposes data to be in ("Sample", "Antigen", "Receptor") order 
    and omits all receptor data that does not pertain to the specified antibody
    """
    data = data.transpose(mode_order[0], mode_order[1], mode_order[2])
    data_receptors = data.Receptor.values
    wanted_receptors = [x for x in data_receptors if x.startswith(abs)] + \
                       [x for x in data_receptors if (x.startswith("FcR") and x != "FcRalpha")]
    data[np.where(data == np.inf)] = np.nan
    data[np.where(data == -np.inf)] = np.nan
    return data.sel(Receptor=wanted_receptors)

def assemble_Kavf(data: xr.DataArray):
    """
    Assemblies fixed affinities matrix for a given dataset
    """
    f = ["IgG1f", "IgG2f", "IgG3f", "IgG4f"]
    receptors = data.Receptor.values
    
    # get known affinities
    affinities = human_affinity()

    # assemble matrix
    data_placeholder = np.full((len(receptors), len(absf)), 10)
    Kav = xr.DataArray(data_placeholder, coords=[receptors, absf], dims=["Receptor", "Abs"])
    
    # separate into list of fc and igg receptors
    fc = [x for x in receptors if (re.match("fc[gr]*", x, flags=re.IGNORECASE) and x != "FcRalpha")]
    igg = [x for x in receptors if (re.match("^igg", x, flags=re.IGNORECASE))]

    # fill in all IgG - IgG pair affinity values
    for ab in absf:
        for ig in igg:
            if (ab == ig or ab[:-1] == ig):
                Kav.loc[dict(Receptor=ig, Abs=ab)] = initial_affinity

    # fill in remaining affinity values
    for ab in absf:
        for r in fc:
            if (ab in f):
                if (re.match("fc[gr]*3", r, flags=re.IGNORECASE)):
                    Kav.loc[dict(Receptor=r, Abs=ab)] = 10
                else:
                    affinity = get_affinity(affinities, r, ab[:-1])
                    Kav.loc[dict(Receptor=r, Abs=ab)] = affinity
            else:
                affinity = get_affinity(affinities, r, ab)
                Kav.loc[dict(Receptor=r, Abs=ab)] = affinity
    
    Kav[np.where(Kav==0.0)] = 10
    return Kav
