from os.path import join, dirname
path_here = dirname(dirname(__file__))

from import_kaplonek import load_file, importSpaceX, cubeSpaceX, flattenSpaceX, importMGH, cubeMGH,flattenMGH

def test_files():
    """ Test that files are successfully loaded. """
    load_file("SpaceX_Sero.Data")
    importSpaceX()
    cubeSpaceX()
    flattenSpaceX()
    importMGH()
    cubeMGH()
    flattenMGH()
