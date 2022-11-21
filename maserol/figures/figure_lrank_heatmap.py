from maserol.core import optimizeLoss, initializeParams, flattenParams
from maserol.heatmap import plotHeatmaps
from maserol.preprocess import prepare_data

from tensordata.zohar import data as zohar


def makeFigure():
    data = prepare_data(zohar())
    x_opt, _ = optimizeLoss(data, lrank=True)
    return plotHeatmaps(data, x_opt, lrank=True)[0]