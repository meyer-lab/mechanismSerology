import pytest
from ..preprocess import *
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data3D as zohar
from tensordata.kaplonek import MGH, SpaceX

@pytest.mark.parametrize("data", [atyeo(xarray=True),
                                  zohar(xarray=True),
                                  MGH(xarray = True),
                                  SpaceX(xarray = True)])
def test_prepare_data(data):
    """ Test prepare_data() can rotate dims and remove irrelevant receptors """
    cube = prepare_data(data)
    assert cube.dims[1] == "Receptor"
    assert all([x in cube.Receptor for x in ["IgG1"]])
    assert all([x not in cube.Receptor for x in ["IgA", "IgA1", "FcRalpha", "IgM", "C1q", "SNA", "ADCC"]])

@pytest.mark.parametrize("data", [zohar(xarray=True),
                                  MGH(xarray = True),
                                  SpaceX(xarray = True)])
def test_assembleKav(data):
    data = prepare_data(data)
    Ka = assembleKav(data, ab_types=HIgGFs)
    assert Ka.sel(Receptor="IgG1", Abs="IgG2") == 10
    assert Ka.sel(Receptor="IgG3", Abs="IgG3f") == 1e8
    assert Ka.sel(Receptor="FcR3A", Abs="IgG2f") == 10
    assert Ka.sel(Receptor="FcR3A", Abs="IgG2") == 7e4
    assert Ka.sel(Receptor="FcR2A", Abs="IgG4") == 2e5
    assert all([~np.all(Ka.sel(Receptor=r) <= 10) for r in Ka.Receptor])
