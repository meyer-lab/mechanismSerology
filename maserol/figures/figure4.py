from tensordata.atyeo import data as atyeo
from ..predictAbundKa import *
from ..plots import makeComponentPlot

def makeFigure():
    """ """
    data = atyeo()
    opt = optimize_lossfunc(data.tensor, n_ab=2)
    R_subj, R_Ag, Ka = reshapeParams(opt, data.tensor)
    return makeComponentPlot([R_subj, Ka, R_Ag], data.axes)