""" Import binding affinities. """
import re
from pathlib import Path
from typing import Collection

import yaml
import numpy as np
import pandas as pd
import xarray as xr

class AffinityNotFoundException(Exception):
    def __init__(self, receptor: str, ab_type: str):
        super().__init__(f"Receptor: {receptor}, Antibody Type: {ab_type}")

class HDataArray(xr.DataArray):
    """
    Hashable data array

    For allowing DataArray to be passed as a static argument to jax.jit
    """
    def __hash__(self):
        return hash(str(self.values))
    
    __slots__ =  () # avoid future warning from xarray

PROJ_DIR = Path(__file__).parent
CONFIGS_PATH = PROJ_DIR / "data_configs.yaml"

HIgGs = ("IgG1", "IgG2", "IgG3", "IgG4")
HIgGFs = ("IgG1", "IgG1f", "IgG2", "IgG2f", "IgG3", "IgG3f", "IgG4", "IgG4f")

DEFAULT_AB_TYPES = HIgGFs

def prepare_data(data: xr.DataArray, remove_rcp=None, data_id=None):
    """
    Transposes data to be in ("Sample", "Receptor", "Antigen") order
    and omits any data not pertaining to IgG or FcgR.

    data_id is declared in data_configs.yaml. Pass the data_id into this
    function to have the operations for this data_id in data_configs.yaml be
    applied to the dataset 
    """
    assert len(data.dims) == 3, "Data must be 3 dimensional"
    # Make the modes in the right order
    if "Subject" in data.dims:
        data = data.rename({"Subject": "Sample"})
    data = data.transpose("Sample", "Receptor", "Antigen")

    wanted_receptors = data.Receptor.values
    if data_id is None:
        # Receptors: only keep those with "IgGx" or "FcgRx"
        wanted_receptors = [x for x in wanted_receptors if re.match("^igg", x, flags=re.IGNORECASE)] + \
                           [x for x in wanted_receptors if re.match("fc[gr]*", x, flags=re.IGNORECASE) and x != "FcRalpha"]
        if remove_rcp != None:
            for r in remove_rcp:
                wanted_receptors.remove(r)
        data = data.sel(Receptor=wanted_receptors)
    else:
        with open(CONFIGS_PATH, "r") as f:
            configs = yaml.load(f, yaml.Loader)
        config = configs.get(data_id)
        if config is not None:
            trans = config.get("translations")
            if trans is not None:
                rcp_trans = trans.get("rcp", {})
                ag_trans = trans.get("ag", {})
                data["Receptor"] = [rcp_trans.get(rcp, rcp) for rcp in data["Receptor"].values]
                data["Antigen"] = [ag_trans.get(ag, ag) for ag in data["Antigen"].values]
            include = config.get("include")
            if include is not None:
                rcp_include = include.get("rcp")
                ag_include = include.get("ag")
                if rcp_include is not None:
                    data = data.sel(Receptor=rcp_include)
                if ag_include is not None:
                    data = data.sel(Antigen=ag_include)

    # Antigens: remove those with all missing values
    missing_ag = []
    for antigen in data.Antigen:
        if not np.any(np.isfinite(data.sel(Antigen=antigen))):  # only nan values for antigen
            missing_ag.append(antigen.values)
    data = data.drop_sel(Antigen=missing_ag)
    return HDataArray(data)


def get_affinity(rcp: str, ab_type: str) -> float:
    """ 
    Given a receptor and an antibody, returns their affinity value in human.
    """
    df = pd.read_csv(PROJ_DIR / "data" / "human-affinities.csv", 
                     delimiter=",", comment="#", index_col=0)

    # figure out of receptor uses iii or 1,2,3 system
    x = re.search("3|2|1|i+", rcp, flags=re.IGNORECASE)
    if x is not None: 
        match = x.group()
        if (match == '1' or match.lower() == 'i'):
            num = "I"
        elif (match == '2' or match.lower() == 'ii'):
            num = "II"
        else:
            num = "III"

        # search for receptor match in affinities dataArray
        for r in list(df.index):
            r_regex = "fc[gr]*" + num + rcp[x.end()::]
            if re.match(r_regex, r, flags=re.IGNORECASE):
                return df.at[r,ab_type]
    try:
        return df.at[rcp,ab_type]
    except KeyError:
        raise AffinityNotFoundException(rcp, ab_type)


def assembleKav(data: xr.DataArray, ab_types: Collection=DEFAULT_AB_TYPES) -> xr.DataArray:
    """ Assemble affinity matrix for a given dataset. """
    receptors = data.Receptor.values    # work even when data did not go thru prepare_data()
    igg = [x for x in receptors if (re.match("^igg", x, flags=re.IGNORECASE))]

    # assemble matrix
    Kav = xr.DataArray(np.full((receptors.size, len(ab_types)), 10),
                       coords=[receptors, list(ab_types)],
                       dims=["Receptor", "Abs"])

    # fill in all IgG - IgG pair affinity values
    for ab in ab_types:
        for ig in igg:
            if (ab == ig or ab[:-1] == ig):
                Kav.loc[dict(Receptor=ig, Abs=ab)] = 1e7      # default affinity for anti-IgGx Ab

    # fill in remaining affinity values
    for ab in ab_types:
        for r in receptors:
            if r in igg:
                continue
            Kav.loc[dict(Receptor=r, Abs=ab)] = get_affinity(r, ab)
    
    Kav[np.where(Kav<10.0)] = 10
    return Kav

def makeRcpAgLabels(data: xr.DataArray):
    data_flat = data.stack(label=["Sample", "Receptor", "Antigen"])["label"]
    return np.array([x.Receptor.values for x in data_flat]), \
           np.array([x.Antigen.values for x in data_flat])

