"""
This is an internal figure which does not necessarily have a place in final
paper.
"""
from tensordata.zohar import data as zohar

from maserol.preprocess import prepare_data
from maserol.scatterplot import plotLRcpO

def makeFigure():
    data = prepare_data(zohar())
    return plotLRcpO(data, "FcR3A")
