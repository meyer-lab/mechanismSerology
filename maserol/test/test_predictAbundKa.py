from ..predictAbundKa import initial_AbundKa, model_lossfunc
from ..data.kaplonek import flattenSpaceX


def test_SpaceX():
    flatCube, _, _ = flattenSpaceX()
    R_guess, Ka_guess = initial_AbundKa(flatCube, 3)
    loss = model_lossfunc(R_guess, Ka_guess, flatCube)
