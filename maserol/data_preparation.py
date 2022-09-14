""" Import binding affinities. """

from os.path import join, dirname
import numpy as np
import pandas as pd
import xarray as xr
import re

path_here = dirname(dirname(__file__))
mode_order = ["Sample", "Receptor", "Antigen"]

def get_affinity(receptor, abs):
    """ 
    Given a receptor and an antibody, returns their affinity value in human.
    """
    df = pd.read_csv(join(path_here, "maserol/data/human-affinities.csv"),
                     delimiter=",", comment="#", index_col=0)
    df.drop(["FcgRIIA-131R", "FcgRIIB-232T", "FcgRIIIA-158F"], inplace=True)

    # figure out of receptor uses iii or 1,2,3 system
    x = re.search("3|2|1|i+", receptor, flags=re.IGNORECASE)
    match = x.group()
    if (match == '1' or match.lower() == 'i'):
        num = "I"
    elif (match == '2' or match.lower() == 'ii'):
        num = "II"
    else:
        num = "III"

    # search for receptor match in affinities dataArray
    for r in list(df.index):
        r_regex = "fc[gr]*" + num + receptor[x.end()::]
        if re.match(r_regex, r, flags=re.IGNORECASE):
            return df.at[r,abs]
    return 0

def prepare_data(data: xr.DataArray, remove_rcp=None, exp=False):
    """
    Transposes data to be in ("Sample", "Antigen", "Receptor") order 
    and omits all receptor data that does not pertain to the specified antibody.
    """
    if exp:
        data = np.exp(data)
    data = data.transpose(mode_order[0], mode_order[1], mode_order[2])
    data_receptors = data.Receptor.values
    wanted_receptors = [x for x in data_receptors if re.match("^igg", x, flags=re.IGNORECASE)] + \
                       [x for x in data_receptors if re.match("fc[gr]*", x, flags=re.IGNORECASE) and x != "FcRalpha"]
    
    # remove receptors specified in 'remove' parameter
    if remove_rcp != None:
        for r in remove_rcp:
            wanted_receptors.remove(r)

    assert np.all(np.isfinite(data)), "In prepare_data(), some entries contain infinity or NaN."
    missing_ag = []

    # remove antigens with all missing values
    for antigen in data.Antigen:
        if np.unique(data.sel(Antigen = antigen).values).size == 1: # only nan values for antigen
            missing_ag.append(antigen.values)
    data = data.drop(labels = missing_ag, dim='Antigen')
    return data.sel(Receptor=wanted_receptors)

def assemble_Kav(data: xr.DataArray, fucose=True):
    """
    Assemblies fixed affinities matrix for a given dataset.
    """
    absf = ["IgG1", "IgG2", "IgG3", "IgG4"]
    if fucose:
        absf = ["IgG1", "IgG1f", "IgG2", "IgG2f", "IgG3", "IgG3f", "IgG4", "IgG4f"]
    f = ["IgG1f", "IgG2f", "IgG3f", "IgG4f"]
    receptors = data.Receptor.values

    # assemble matrix
    Kav = xr.DataArray(np.full((len(receptors), len(absf)), 10),
                       coords=[receptors, absf],
                       dims=["Receptor", "Abs"])
    
    # separate into list of fc and igg receptors
    fc = [x for x in receptors if (re.match("fc[gr]*", x, flags=re.IGNORECASE) and x != "FcRalpha")]
    igg = [x for x in receptors if (re.match("^igg", x, flags=re.IGNORECASE))]

    # fill in all IgG - IgG pair affinity values
    for ab in absf:
        for ig in igg:
            if (ab == ig or ab[:-1] == ig):
                Kav.loc[dict(Receptor=ig, Abs=ab)] = 10**8      # default affinity for anti-IgGx Ab

    # fill in remaining affinity values
    for ab in absf:
        for r in fc:
            if (ab in f):
                if (re.match("fc[gr]*3", r, flags=re.IGNORECASE)):
                    Kav.loc[dict(Receptor=r, Abs=ab)] = 10
                else:
                    affinity = get_affinity(r, ab[:-1])
                    Kav.loc[dict(Receptor=r, Abs=ab)] = affinity
            else:
                affinity = get_affinity(r, ab)
                Kav.loc[dict(Receptor=r, Abs=ab)] = affinity
    
    Kav[np.where(Kav<10.0)] = 10
    return Kav
