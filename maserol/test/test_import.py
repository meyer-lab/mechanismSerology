"""
Unit test file.
"""
from ..data.alter import importLuminex, createCube
from ..data.kaplonek import importSpaceX, flattenSpaceX, cubeSpaceX, importMGH, cubeMGH, flattenMGH


def test_imports_alter():
    """ Test that files are successfully loaded. """
    importLuminex()
    createCube()


def test_imports_kaplonek():
    """ Test that files are successfully loaded. """
    importSpaceX()
    cubeSpaceX()
    flattenSpaceX()
    importMGH()
    cubeMGH()
    flattenMGH()
