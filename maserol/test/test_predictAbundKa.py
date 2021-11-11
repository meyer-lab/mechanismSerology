from ..predictAbundKa import initial_AbundKa, optimize_lossfunc
from ..data.kaplonek import cubeSpaceX


def test_SpaceX():
    cube = cubeSpaceX()
    R_subj_guess, R_Ag_guess, Ka_guess = initial_AbundKa(cube, 1)
    RKa_opt = optimize_lossfunc(cube, 1)
    #compare(RKa_opt, cube)
