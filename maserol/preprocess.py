""" Import binding affinities. """
import re
from pathlib import Path
from typing import Collection

import numpy as np
import pandas as pd
import xarray as xr

from tensordata.kaplonek import MGH4D


class AffinityNotFoundException(Exception):
    def __init__(self, ligand: str, receptor: str):
        super().__init__(f"Ligand: {ligand}, Receptor: {receptor}")


PROJ_DIR = Path(__file__).parent

# standard Ab subclasses
HIgGs = ("IgG1", "IgG2", "IgG3", "IgG4")
# Ab subclasses with fucosylated counterparts
HIgGFs = ("IgG1", "IgG1f", "IgG2", "IgG2f", "IgG3", "IgG3f", "IgG4", "IgG4f")
# IgG1 and IgG3 fucosylated
IgG1_3 = ("IgG1", "IgG1f", "IgG3", "IgG3f")

DEFAULT_RCPS = IgG1_3


def prepare_data(data: xr.DataArray, ligs=None):
    """
    Reshapes data into matrix of size (n_cplx, n_lig), where n_cplx = n_sample x
    n_ag - missing.
    """
    assert len(data.dims) == 3, "Data must be 3 dimensional"
    # Antigens: remove those with all missing values
    missing_ag = []
    for antigen in data.Antigen:
        if not np.any(
            np.isfinite(data.sel(Antigen=antigen))
        ):  # only nan values for antigen
            missing_ag.append(antigen.values)
    if "Subject" in data.dims:
        data = data.rename({"Subject": "Sample"})
    data = (
        data.drop_sel(Antigen=missing_ag)
        .rename({"Receptor": "Ligand"})
        .stack(Complex=("Sample", "Antigen"))
        .transpose("Complex", "Ligand")
    )
    ligs = ligs or [
        lig
        for lig in data.Ligand.values
        if re.match("^igg|fc[gr]*", lig, flags=re.IGNORECASE) and lig != "FcRalpha"
    ]
    data = data.sel(
        Ligand=ligs,
    )
    return data[np.all((data.values != 0) & ~np.isnan(data.values), axis=1)]


def get_affinity(lig: str, rcp: str) -> float:
    """
    Given a ligand and receptors, retreives their affinity value.
    """
    df = pd.read_csv(
        PROJ_DIR / "data" / "human-affinities.csv",
        delimiter=",",
        comment="#",
        index_col=0,
    )

    if re.search("^IgG[1-4]$", lig):
        # subclass-specific detection reagent
        return df.at[lig, rcp]

    # figure out of receptor uses iii or 1,2,3 system
    x = re.search("3|2|1|i+", lig, flags=re.IGNORECASE)
    if x is not None:
        match = x.group()
        if match == "1" or match.lower() == "i":
            num = "I"
        elif match == "2" or match.lower() == "ii":
            num = "II"
        else:
            num = "III"

        # search for receptor match in affinities dataArray
        for r in list(df.index):
            r_regex = "fc[gr]*" + num + lig[x.end() : :]
            if re.match(r_regex, r, flags=re.IGNORECASE):
                return df.at[r, rcp]
    try:
        return df.at[lig, rcp]
    except KeyError:
        raise AffinityNotFoundException(lig, rcp)


def assemble_Ka(data: xr.DataArray, rcps: Collection = DEFAULT_RCPS) -> xr.DataArray:
    """Assemble affinity matrix for a given dataset."""
    ligs = data.Ligand.values  # work even when data did not go thru prepare_data()

    # assemble matrix
    Ka = xr.DataArray(
        np.full((ligs.size, len(rcps)), 10),
        coords=[ligs, list(rcps)],
        dims=["Ligand", "Receptor"],
    )

    # fill in remaining affinity values
    for rcp in rcps:
        for l in ligs:
            Ka.loc[dict(Ligand=l, Receptor=rcp)] = get_affinity(l, rcp)

    Ka.values[np.where(Ka.values < 10.0)] = 10
    Ka.values = Ka.values.astype("float")
    return Ka


def assemble_options(
    data: xr.DataArray,
    rcps=DEFAULT_RCPS,
    IgG_L0: float = 1e-9,
    Fc_L0: float = 1e-9,
    IgG_KxStar: float = 1e-12,
    Fc_KxStar: float = 1e-12,
    IgG_logistic: bool = True,
):
    """
    Helper function for constructing parameters used in optimization.

    Args:
        data, rcps: see `optimize_loss`.
        IgG_L0: L0 to use for IgG (subclass) detection reagents
        Fc_L0: L0 to use for FcR detection reagents
        IgG_KxStar: KxStar to use for IgG detection reagents
        Fc_KxStar: KxStar to use for FcR detection reagents

    Returns:
      Dictionary with `optimize_loss` params.
    """
    n_lig = data.sizes["Ligand"]
    n_rcp = len(rcps)
    IgG_re = re.compile("^IgG[1-4]$")
    L0 = np.full(n_lig, 1e-9)
    KxStar = np.full(n_lig, 1e-12)
    f = np.full(n_lig, 4)
    for i in range(n_lig):
        if IgG_re.search(data.Ligand.values[i]):
            L0[i] = IgG_L0
            KxStar[i] = IgG_KxStar
        else:
            L0[i] = Fc_L0
            KxStar[i] = Fc_KxStar
    for i in range(n_lig):
        if IgG_re.search(data.Ligand.values[i]):
            f[i] = 2
    logistic_ligands = np.full((n_lig, n_rcp), False)
    if IgG_logistic:
        for i in range(n_lig):
            logistic_ligands[i] = np.array(
                [data.Ligand.values[i] in rcp for rcp in rcps]
            )

    return {
        "L0": L0,
        "KxStar": KxStar,
        "f": f,
        "rcps": rcps,
        "logistic_ligands": logistic_ligands,
        "fitKa": False,
    }


def get_kaplonek_mgh_data():
    mgh_4d = MGH4D()["Serology"]

    tensors = [10 ** mgh_4d.isel(Time=i) for i in range(mgh_4d.sizes["Time"])]

    for tensor in tensors:
        tensor.values[tensor.values == 1] = 0
        tensor.coords["Subject"] = [
            f"{sample}_{tensor.Time.values}" for sample in tensor.Subject.values
        ]

    tensors = [
        prepare_data(tensor, ligs=["IgG1", "IgG3", "FcR2A", "FcR2B", "FcR3A", "FcR3B"])
        for tensor in tensors
    ]
    return xr.concat(tensors, dim="Complex")
