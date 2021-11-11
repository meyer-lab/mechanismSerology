from ..import_kaplonek import *
from os.path import join, dirname
path_here = dirname(dirname(__file__))


def test_imports():
    """ Test that files are successfully loaded. """
    load_file("SpaceX_Sero.Data")
    importSpaceX()
    cubeSpaceX()
    flattenSpaceX()
    importMGH()
    cubeMGH()
    flattenMGH()
