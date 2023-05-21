import pytest
import numpy as np
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data as zohar
from tensordata.kaplonek import MGH4D, SpaceX4D
from ..preprocess import prepare_data, assembleKav, HIgGFs


@pytest.mark.parametrize("data", [atyeo(),
                                  zohar(),
                                  MGH4D()["Serology"].stack(Sample = ("Subject", "Time")),
                                  SpaceX4D().stack(Sample = ("Subject", "Time"))])
def test_prepare_data(data):
    """ Test prepare_data() can rotate dims and remove irrelevant receptors """
    cube = prepare_data(data)
    assert cube.dims[1] == "Receptor"
    assert all([x in cube.Receptor for x in ["IgG1"]])
    assert all([x not in cube.Receptor for x in ["IgA", "IgA1", "FcRalpha", "IgM", "C1q", "SNA", "ADCC"]])

@pytest.mark.parametrize("data", [zohar(),
                                  MGH4D()["Serology"].stack(Sample = ("Subject", "Time")),
                                  SpaceX4D().stack(Sample = ("Subject", "Time"))])
def test_assembleKav(data):
    data = prepare_data(data)
    Ka = assembleKav(data, ab_types=HIgGFs, newAff=False)
    assert Ka.sel(Receptor="IgG1", Abs="IgG2") == 10
    assert Ka.sel(Receptor="IgG3", Abs="IgG3f") == 1e7
    assert Ka.sel(Receptor="FcR3A", Abs="IgG2") == 7e4
    assert Ka.sel(Receptor="FcR2A", Abs="IgG4") == 2e5
    assert all([~np.all(Ka.sel(Receptor=r) <= 10) for r in Ka.Receptor])
    Ka = assembleKav(data, ab_types=HIgGFs, newAff=True)
    assert Ka.sel(Receptor="FcR3A", Abs="IgG1") > 1e6

