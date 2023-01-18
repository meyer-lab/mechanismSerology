from tensordata.zohar import data as zohar

from ..core import optimizeLoss
from ..heatmap import plotHeatmaps
from ..preprocess import prepare_data


def makeFigure():
    data = prepare_data(zohar())
    x_opt, _ = optimizeLoss(data, lrank=True)
    return plotHeatmaps(data, x_opt, lrank=True)[0]