from os.path import join, dirname
path_here = dirname(dirname(__file__))

from ..import_kaplonek import importSpaceX, cubeSpaceX, flattenSpaceX, importMGH, cubeMGH,flattenMGH

def test_files():
    """ Test that files are successfully loaded. """
    importSpaceX()
    cubeSpaceX()
    flattenSpaceX()
    importMGH()
    cubeMGH()
    flattenMGH()
