from tensordata.atyeo import data as atyeo
from tensorpack import perform_CP
from ..plots import makeComponentPlot

def makeFigure():
    """ Generate heatmap plots for each input dimension by component"""
    data = atyeo()
    tfac = perform_CP(tOrig=data.tensor)
    return makeComponentPlot([tfac.factors[0], tfac.factors[1], tfac.factors[2]], atyeo.axes)

