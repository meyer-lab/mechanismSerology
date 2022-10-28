"""
This is an internal figure which does not necessarily have a place in final
paper.
"""
from tensordata.zohar import data3D as zohar

from maserol.core import prepare_data
from maserol.scatterplot import plotLORecO

def makeFigure():
    data = prepare_data(zohar(xarray=True, logscale=False))
    return plotLORecO(data, "FcR2B", metric="rtot")
