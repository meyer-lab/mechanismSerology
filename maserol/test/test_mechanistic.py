import pytest
from ..mechanistic import *
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data3D as zohar

@pytest.mark.parametrize("n_ab", [1, 2, 3])
def test_initialize(n_ab):
    """ Test initializeParams() work correctly """
    cube = atyeo(xarray=True)
    n_samp, n_recp, n_ag = cube.shape
    ps = initializeParams(cube, lrank=True, fitKa=False, n_ab=n_ab)  # should return subj, ag
    assert len(ps) == 2
    assert ps[1].shape == (n_ag, n_ab)
    ps = initializeParams(cube, lrank=False, fitKa=True, n_ab=n_ab)  # should return abund, Ka
    assert len(ps) == 2
    assert ps[0].shape == (n_samp * n_ag, n_ab)
    assert ps[1].shape == (n_recp, n_ab)



@pytest.mark.parametrize("n_ab", [1, 2, 3])
def test_fit_Ka(n_ab):

    pass






def test_fit_mean():
    pass

def test_fit_rtot():
    pass

def test_fit_r():
    pass

