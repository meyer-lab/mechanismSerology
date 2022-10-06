import pytest
from ..preprocess import *
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data3D as zohar
from tensordata.alter import data as alter
from tensordata.kaplonek import MGH, SpaceX

@pytest.mark.parametrize("data", [atyeo(xarray=True),
                                  zohar(xarray=True),
                                  alter(xarray = True)["Fc"],
                                  MGH(xarray = True),
                                  SpaceX(xarray = True)])
def test_prepare_data(data):
    """ Test prepare_data() can rotate dims and remove irrelevant receptors """
    cube = prepare_data(data)
    assert cube.dims[1] == "Receptor"
    assert all([x in cube.Receptor for x in ["IgG1"]])
    assert all([x not in cube.Receptor for x in ["IgA", "IgA1", "FcRalpha", "IgM", "C1q", "SNA", "ADCC"]])
