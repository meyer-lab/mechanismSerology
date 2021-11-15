"""
Unit test file.
"""

def test_imports_alter():
    """ Test that files are successfully loaded. """
    from ..data.alter import importLuminex, createCube
    importLuminex()
    createCube()


def test_imports_kaplonek():
    """ Test that files are successfully loaded. """
    from ..data.kaplonek import importSpaceX, flattenSpaceX, cubeSpaceX, importMGH, cubeMGH, flattenMGH
    importSpaceX()
    cubeSpaceX()
    flattenSpaceX()
    importMGH()
    cubeMGH()
    flattenMGH()


def test_import_atyeo():
    """ Test that files are successfully loaded. """
    from ..data.atyeo import load_file, getAxes, createCube, flattenCube
    load_file("atyeo_covid")
    getAxes()
    createCube()
    flattenCube()