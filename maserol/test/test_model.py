"""
Test any functionality that is related to the binding model
"""

from ..model import human_affinity

def test_import_affinity():
    """ Test that affinity file is loaded correctly. """
    df = human_affinity()
    assert df["IgG1"]["FcgRIIB"] > 0
    assert df["IgG2b"]["FcgRIV"] > 0