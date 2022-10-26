from maserol.core import optimizeLoss, initializeParams, flattenParams
from maserol.heatmap import plotHeatmaps
from maserol.preprocess import prepare_data

from tensordata.zohar import data3D as zohar


def makeFigure():
    data = prepare_data(zohar(xarray=True, logscale=False))
    x_opt, _ = optimizeLoss(data, lrank=True)
    return plotHeatmaps(data, x_opt, lrank=True)[0]