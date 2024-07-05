"""Import binding affinities."""

import re
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
import xarray as xr

from maserol.concentration import get_Fc_detection_molarity


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
        raise AffinityNotFoundException(lig, rcp) from None


def assemble_Ka(
    ligs: Iterable, rcps: Iterable = DEFAULT_RCPS, logistic_ligands=None
) -> xr.DataArray:
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
    for rcp in rcps:
        for lig in ligs:
            Ka.loc[dict(Ligand=lig, Receptor=rcp)] = get_affinity(lig, rcp)

    Ka.values[np.where(Ka.values < 10.0)] = 10
    Ka.values = Ka.values.astype("float")
    return Ka


def assemble_options(
    data: xr.DataArray,
    rcps=DEFAULT_RCPS,
    FcR_KxStar: float = 1e-12,
    FcR_f: int = 4,
    FcR_conc_ug_mL: float = 1,
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
    KxStar = np.full(n_lig, FcR_KxStar)
    f = np.full(n_lig, FcR_f)

    logistic_ligands = np.full((n_lig, n_rcp), False)
    for i in range(n_lig):
        logistic_ligands[i] = np.array(
            # rcp name includes ligand name (e.g. IgG1 ligand includes IgG1
            # and IgG1f receptors)
            [data.Ligand.values[i] in rcp for rcp in rcps]
        )
    logistic_ligs = logistic_ligand_map(logistic_ligands)
    for lig in data.Ligand.values[logistic_ligs]:
        assert "IgG" in lig
    multivalent_ligs = ~logistic_ligs
    n_mvl = np.count_nonzero(multivalent_ligs)
    FcRs = data.Ligand.values[multivalent_ligs]
    for r in FcRs:
        assert "Fc" in r
    KxStar = np.full(n_mvl, FcR_KxStar)
    f = np.full(n_mvl, FcR_f)
    L0 = np.zeros(n_mvl)
    for i, r in enumerate(FcRs):
        L0[i] = get_Fc_detection_molarity(r.split("-")[0], FcR_conc_ug_mL)
    opts = {
        "L0": L0,
        "KxStar": KxStar,
        "f": f,
        "rcps": rcps,
        "logistic_ligands": logistic_ligands,
    }
    return opts


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


def compute_fucose_ratio(Rtot: pd.DataFrame) -> pd.Series:
    assert np.all(Rtot.columns.values == np.array(IgG1_3))
    fucose = (
        (Rtot["IgG1f"] + Rtot["IgG3f"])
        / (Rtot["IgG1"] + Rtot["IgG1f"] + Rtot["IgG3"] + Rtot["IgG3f"])
        * 100
    )
    fucose.name = "fucose_inferred"
    return fucose
