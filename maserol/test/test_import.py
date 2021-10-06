"""
Unit test file.
"""
from ..import_alter import load_file, importLuminex, createCube


def test_files():
    """ Test that files are successfully loaded. """
    load_file("data-luminex")
    importLuminex()
    createCube()
