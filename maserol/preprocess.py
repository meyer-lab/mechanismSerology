""" Import binding affinities. """

from os.path import join, dirname
import numpy as np
import pandas as pd
import xarray as xr
import re

path_here = dirname(dirname(__file__))


def prepare_data(data: xr.DataArray, remove_rcp=None):
    """
    Transposes data to be in ("Sample", "Receptor", "Antigen") order
    and omits any data not pertaining to IgG or FcgR.
    """
    # Make the modes in the right order
    data = data.transpose("Sample", "Receptor", "Antigen")

    # Receptors: only keep those with "IgGx" or "FcgRx"
    data_receptors = data.Receptor.values
    wanted_receptors = [x for x in data_receptors if re.match("^igg", x, flags=re.IGNORECASE)] + \
                       [x for x in data_receptors if re.match("fc[gr]*", x, flags=re.IGNORECASE) and x != "FcRalpha"]
    if remove_rcp != None:
        for r in remove_rcp:
            wanted_receptors.remove(r)
    data = data.sel(Receptor=wanted_receptors)

    # Antigens: remove those with all missing values
    missing_ag = []
    for antigen in data.Antigen:
        if not np.any(np.isfinite(data.sel(Antigen=antigen))):  # only nan values for antigen
            missing_ag.append(antigen.values)
    data = data.drop_sel(Antigen=missing_ag)
    return data


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


def assembleKav(data: xr.DataArray, fucose=True):
    """ Assemble affinity matrix for a given dataset. """
    absf = ["IgG1", "IgG2", "IgG3", "IgG4"] if not fucose else \
        ["IgG1", "IgG1f", "IgG2", "IgG2f", "IgG3", "IgG3f", "IgG4", "IgG4f"]

    receptors = data.Receptor.values    # work even when data did not go thru prepare_data()
    igg = [x for x in receptors if (re.match("^igg", x, flags=re.IGNORECASE))]
    fc = [x for x in receptors if (re.match("fc[gr]*", x, flags=re.IGNORECASE) and x != "FcRalpha")]

    # assemble matrix
    Kav = xr.DataArray(np.full((len(igg+fc), len(absf)), 10),
                       coords=[igg+fc, absf],
                       dims=["Receptor", "Abs"])

    # fill in all IgG - IgG pair affinity values
    for ab in absf:
        for ig in igg:
            if (ab == ig or ab[:-1] == ig):
                Kav.loc[dict(Receptor=ig, Abs=ab)] = 10**8      # default affinity for anti-IgGx Ab

    # fill in remaining affinity values
    for ab in absf:
        for r in fc:
            if not (ab[-1]=='f' and re.match("fc[gr]*3", r, flags=re.IGNORECASE)):
                affinity = get_affinity(r, ab[:4])
                Kav.loc[dict(Receptor=r, Abs=ab)] = affinity
    
    Kav[np.where(Kav<10.0)] = 10
    return Kav

def makeRcpAgLabels(data: xr.DataArray):
    data_flat = data.stack(label=["Sample", "Receptor", "Antigen"])["label"]
    return np.array([x.Receptor.values for x in data_flat]), \
           np.array([x.Antigen.values for x in data_flat])

def normalize_subj_ag(subj, ag, n_ab, whole=True):
    """
        Normalizes antigen matrix in factor plotting.
        If 'whole' is False, normalizes antigen matrix by columns.
    """
    if (whole):
        ag /= ag.max()
        subj *= ag.max()
    else:
        for i in range(n_ab):
            max = ag[:,i].max()
            ag = ag.at[:,i].set(ag[:,i] / max)
            subj = subj.at[:,i].set(subj[:,i] * max)
    return subj, ag
