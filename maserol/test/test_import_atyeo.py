"""
Unit test file.
"""
from ..import_atyeo import load_file, getAxes, createCube, flattenCube


def test_files():
    """ Test that files are successfully loaded. """
    load_file("atyeo_covid")
    getAxes()
    createCube()
    flattenCube()