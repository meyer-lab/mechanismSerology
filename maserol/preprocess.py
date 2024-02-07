""" Import binding affinities. """

import re
from pathlib import Path
from typing import Iterable, List, Union

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
            print(r_regex)
            if re.match(r_regex, r, flags=re.IGNORECASE):
                return df.at[r, rcp]
    try:
        return df.at[lig, rcp]
    except KeyError:
        raise AffinityNotFoundException(lig, rcp)


def assemble_Ka(
    ligs: Iterable, rcps: Iterable = DEFAULT_RCPS, logistic_ligands=None
) -> xr.DataArray:
    """Assemble affinity matrix for a given dataset."""

    assert len(ligs.shape) == 1, "ligs should be 1d"
    if logistic_ligands is not None:
        ligs = ligs[~logistic_ligand_map(logistic_ligands)]

    # assemble matrix
    Ka = xr.DataArray(
        np.full((ligs.size, len(rcps)), 10, dtype=float),
        coords=[ligs, list(rcps)],
        dims=["Ligand", "Receptor"],
    )

    # fill in remaining affinity values
    for r in rcps:
        for l in ligs:
            Ka.loc[dict(Ligand=l, Receptor=r)] = get_affinity(l, r)

    Ka.values[np.where(Ka.values < 10.0)] = 10
    Ka.values = Ka.values.astype("float")
    return Ka


def assemble_options(
    data: xr.DataArray,
    rcps=DEFAULT_RCPS,
    IgG_L0: float = 1e-9,
    FcR_L0: float = 1e-9,
    IgG_KxStar: float = 1e-12,
    FcR_KxStar: float = 1e-12,
    IgG_logistic: bool = True,
    IgG_f: int = 2,
    FcR_f: int = 4,
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
    is_IgG = np.array(
        [bool(re.search("^IgG[1-4]$", lig)) for lig in data.Ligand.values]
    )
    L0 = np.full(n_lig, FcR_L0)
    L0[is_IgG] = IgG_L0
    KxStar = np.full(n_lig, FcR_KxStar)
    KxStar[is_IgG] = IgG_KxStar
    f = np.full(n_lig, FcR_f)
    f[is_IgG] = IgG_f
    logistic_ligands = np.full((n_lig, n_rcp), False)

    if IgG_logistic:
        for i in range(n_lig):
            logistic_ligands[i] = np.array(
                # rcp name includes ligand name (e.g. IgG1 ligand includes IgG1
                # and IgG1f receptors)
                [data.Ligand.values[i] in rcp for rcp in rcps]
            )
        mvl = ~logistic_ligand_map(logistic_ligands)
        L0 = L0[mvl]
        KxStar = KxStar[mvl]
        f = f[mvl]
    return {
        "L0": L0,
        "KxStar": KxStar,
        "f": f,
        "rcps": rcps,
        "logistic_ligands": logistic_ligands,
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


def logistic_ligand_map(logistic_ligands: np.ndarray) -> int:
    return np.sum(logistic_ligands, axis=1) != 0


def n_logistic_ligands(logistic_ligands: np.ndarray) -> int:
    return np.sum(logistic_ligand_map(logistic_ligands))


def Rtot_to_xarray(Rtot: np.ndarray, data: xr.DataArray, rcps: List):
    return xr.DataArray(
        Rtot,
        coords={
            "Complex": data.Complex,
            "Receptor": rcps,
        },
        dims=["Complex", "Receptor"],
    )


def Rtot_to_df(
    Rtot: Union[xr.DataArray, np.ndarray], data: xr.DataArray = None, rcps: List = None
):
    if isinstance(Rtot, np.ndarray):
        assert data is not None, "data required if Rtot is np array"
        assert rcps is not None, "rcps required if Rtot is np array"
        Rtot = Rtot_to_xarray(Rtot, data, rcps)
    Rtot_df = Rtot.to_dataframe(name="Abundance").drop(columns=["Antigen", "Sample"])
    Rtot_df = Rtot_df.reset_index(level="Receptor").pivot(columns="Receptor")
    Rtot_df.columns = [col[1] for col in Rtot_df.columns]
    return Rtot_df


def data_to_df(
    data: xr.DataArray = None,
):
    df = data.to_dataframe(name="Abundance").drop(columns=["Antigen", "Sample"])
    df = df.reset_index(level="Ligand").pivot(columns="Ligand")
    df.columns = [col[1] for col in df.columns]
    return df
