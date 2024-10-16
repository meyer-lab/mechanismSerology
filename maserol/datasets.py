import re
from typing import Literal

import numpy as np
import pandas as pd
import tensordata
import xarray as xr
from tensordata.alter import data as alter
from tensordata.alter import load_file as load_file_alter
from tensordata.kaplonek import MGH4D
from tensordata.kaplonek import load_file as load_file_kaplonek
from tensordata.kaplonekVaccineSA import data as kaplonek_vaccine
from tensordata.zohar import data as zohar

LIG_ORDER = ["IgG1", "IgG3", "FcR2A", "FcR2B", "FcR3A", "FcR3B"]


class Zohar:
    def get_detection_signal(self, select_ligs: bool | None = True) -> xr.DataArray:
        data = zohar(subtract_baseline=True)
        data += np.abs(np.minimum(data.min(dim=["Sample"]), 0))
        data = prepare_data(data)
        if select_ligs:
            data = data.sel(Ligand=LIG_ORDER)
        return data

    def get_metadata(self) -> pd.DataFrame:
        return (
            pd.read_csv(tensordata.zohar.DATA_PATH)
            .rename(columns={"sample_ID": "Sample"})
            .set_index("Sample", drop=True)
        )

    def get_days_binned(self) -> pd.Series:
        days = self.get_metadata()["days"]
        days.name = "days"
        bins = np.arange(0, 40, 5)
        days = pd.cut(days, bins=bins, labels=bins[:-1], right=False)
        return days


class Kaplonek:
    def get_detection_signal(self, select_ligs: bool | None = True) -> xr.DataArray:
        mgh_4d = MGH4D()["Serology"]

        tensors = [10 ** mgh_4d.isel(Time=i) for i in range(mgh_4d.sizes["Time"])]

        for tensor in tensors:
            tensor.values[tensor.values == 1] = 0
            tensor.coords["Subject"] = [
                f"{sample}_{tensor.Time.values}" for sample in tensor.Subject.values
            ]
            antigens = tensor.Antigen.values
            antigens[antigens == "SARS.CoV2_N"] = "SARS.CoV.2_N"
            tensor["Antigen"] = antigens

        tensors = [
            prepare_data(tensor, ligs=LIG_ORDER if select_ligs else None)
            for tensor in tensors
        ]
        return xr.concat(tensors, dim="Complex")

    def get_metadata(self) -> pd.DataFrame:
        df = load_file_kaplonek("MGH_Sero.Meta.data.WHO124")
        return df.drop(columns=["Unnamed: 0"])


class Alter:
    def get_detection_signal(self, select_ligs: bool | None = True) -> xr.DataArray:
        data = alter()["Fc"]
        if select_ligs:
            data = data.sel(
                Receptor=[
                    "IgG1",
                    "IgG3",
                    "FcgRIIa.H131",
                    "FcgRIIa.R131",
                    "FcgRIIb",
                    "FcgRIIIa.F158",
                    "FcgRIIIa.V158",
                    "FcgRIIIb",
                ]
            )
        translate = {
            "FcgRIIa.H131": "FcR2A-131H",
            "FcgRIIa.R131": "FcR2A-131R",
            "FcgRIIb": "FcR2B",
            "FcgRIIIa.V158": "FcR3A-158V",
            "FcgRIIIa.F158": "FcR3A-158F",
            "FcgRIIIb": "FcR3B",
        }
        data = data.assign_coords(
            Receptor=[translate.get(r, r) for r in data.Receptor.values]
        )
        return prepare_data(data)

    def get_glycan_data(self) -> pd.DataFrame:
        glycans = load_file_alter("data-glycan-gp120")
        glycans = glycans.rename(columns={"subject": "Sample"})
        return glycans.set_index("Sample", drop=True)

    def get_fucose_data(self) -> pd.Series:
        glycans = self.get_glycan_data()
        fucose = glycans["F.total"]
        fucose = fucose.dropna()
        fucose = fucose[fucose != 0]
        fucose.name = "fucose_ce"
        return fucose

    def get_metadata(self) -> pd.DataFrame:
        return (
            load_file_alter("meta-subjects")
            .rename(columns={"subject": "Sample"})
            .set_index("Sample")
        )

    def get_subject_class(self) -> pd.Series:
        subject_class = self.get_metadata()["class.etuv"]
        subject_class.name = "class"
        return subject_class

    def get_effector_functions(self) -> pd.DataFrame:
        return (
            load_file_alter("data-function")
            .rename(columns={"subject": "Sample"})
            .set_index("Sample", drop=True)
        )


class KaplonekVaccine:
    def get_detection_signal(self, select_ligs: bool | None = True) -> xr.DataArray:
        data = kaplonek_vaccine()["Luminex"]
        data = prepare_data(data)
        if select_ligs:
            data = data.sel(Ligand=LIG_ORDER)
        return data

    def get_metadata(self) -> pd.DataFrame:
        metadata = (
            kaplonek_vaccine()["Meta"]
            .to_dataframe()
            .reset_index(level="Metadata")
            .pivot(columns="Metadata", values="Meta")
        )
        metadata.columns.name = None
        metadata.index.name = "Sample"
        return metadata


def prepare_data(data: xr.DataArray, ligs: list[str] | Literal["all"] | None = None):
    """
    Reshapes data into matrix of size (n_cplx, n_lig), where n_cplx = n_sample x
    n_ag - missing.
    """
    assert len(data.dims) == 3, "Data must be 3 dimensional"
    # Antigens: remove those with all missing values
    if "Subject" in data.dims:
        data = data.rename({"Subject": "Sample"})
    data = (
        data.rename({"Receptor": "Ligand"})
        .stack(Complex=("Sample", "Antigen"))
        .transpose("Complex", "Ligand")
    )
    if ligs != "all":
        ligs = (
            ligs
            if ligs is not None
            else [
                lig
                for lig in data.Ligand.values
                if re.match("^igg|fc[gr]*", lig, flags=re.IGNORECASE)
                and lig != "FcRalpha"
            ]
        )
        data = data.sel(
            Ligand=ligs,
        )
    return data[np.all(~np.isnan(data.values), axis=1)]


def data_to_df(
    data: xr.DataArray = None,
):
    df = data.to_dataframe(name="Abundance").drop(columns=["Antigen", "Sample"])
    df = df.reset_index(level="Ligand").pivot(columns="Ligand")
    df.columns = [col[1] for col in df.columns]
    return df
