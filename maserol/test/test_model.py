"""
Test any functionality that is related to the binding model
"""
from ..model import human_affinity, assemble_Kav
from tensordata.atyeo import data as atyeo


def test_import_affinity():
    """ Test that affinity file is loaded correctly. """
    df = human_affinity()
    #assert df["IgG1"]["FcgRIIB"] > 0
    #assert df["IgG2b"]["FcgRIV"] > 0


def test_assemble_Kav():
    data = atyeo(xarray = True)
    Kav = assemble_Kav(data)
    abs = ["IgG1", "IgG2", "IgG3", "IgG4"]

    # test only the receptors associated with IgG are in
    not_included = ["FcRalpha", "IgA", "IgA1", "IgA2", "IgM", "IgAI", "IgAII"]
    included = map(lambda x: x.lower(), Kav.Receptor.values)
    for item in not_included:
        assert item.lower() not in list(included)
    
    # IgG - IgG portion
    for ab1 in abs:
        for ab2 in abs:
            if (ab1 == ab2):
                assert Kav.sel(Receptor=ab1, Abs=ab2) == 10**8 # test diagional
            else:
                assert Kav.sel(Receptor=ab1, Abs=ab2) == 0 # test off diagonal
    
    # Various values in other portion 
    assert Kav.sel(Receptor="FcRg2A", Abs="IgG3") == 900000.0
    assert Kav.sel(Receptor="FcRg2b", Abs="IgG2") == 20000.0
    assert Kav.sel(Receptor="FcRg3A", Abs="IgG4") == 200000.0

if __name__ == "__main__":
    test_import_affinity()
    test_assemble_Kav()