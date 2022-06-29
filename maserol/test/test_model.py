"""
Test any functionality that is related to the binding model
"""

from ..model import human_affinity, assemble_Kav
from tensordata.atyeo import data as atyeo


def test_import_affinity():
    """ Test that affinity file is loaded correctly. """
    df = human_affinity()
    assert df["IgG1"]["FcgRIIB"] > 0
    assert df["IgG2b"]["FcgRIV"] > 0


def test_assemble_Kav():
    data = atyeo(xarray = True)
    Kav = assemble_Kav(data)
    # test IgA1 is not in Kav
    # test one or two number
    # assert Kav["IgG3", "FcRg2a"] == 9e5
