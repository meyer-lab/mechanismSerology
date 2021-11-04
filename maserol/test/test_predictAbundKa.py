from ..predictAbundKa import initial_AbundKa, optimize_lossfunc, compare
from ..import_kaplonek import *

def test_SpaceX():
    flatCube, _, _ = flattenSpaceX()
    RKa_combined = initial_AbundKa(flatCube, 1)
    RKa_opt = optimize_lossfunc(RKa_combined, flatCube)
    compare(RKa_opt,flatCube)
