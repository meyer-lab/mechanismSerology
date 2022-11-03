from sklearn.metrics import balanced_accuracy_score

from tensordata.zohar import data3D as zohar

from maserol.core import optimizeLoss, reshapeParams
from maserol.figures.common import getSetup
from maserol.preprocess import prepare_data
from maserol.regression import regression, get_labels_zohar, plot_roc, plot_confusion_matrix

def makeFigure():
    cube = prepare_data(zohar(xarray=True, logscale=False))
    x_opt_lrank, _ = optimizeLoss(cube, lrank=True)
    sample, ag = reshapeParams(x_opt_lrank, cube, lrank=True)
    labels, label_encoder = get_labels_zohar()
    y_pred, model, x, y = regression(sample, labels, scale_x=0, l1_ratio=0)
    axes, fig = getSetup((10, 4), (1, 2))
    ax = plot_roc(x, y, model, label_encoder, axes[0])
    ax.set(title="ROC Curves for Multi-Class Prediction (OVR)")
    ax = plot_confusion_matrix(x, y, model, label_encoder, axes[1])
    ax.set(title="Confusion Matrix for Multi-Class Prediction")
    fig.subplots_adjust(bottom=0.1)
    fig.text(0, 0, f"Balanced accuracy score {round(balanced_accuracy_score(y, y_pred), 2)}")
    return fig
