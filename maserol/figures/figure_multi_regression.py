import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

from tensordata.zohar import data as zohar

from maserol.core import optimize_loss, reshape_params
from maserol.figures.common import getSetup
from maserol.preprocess import prepare_data
from maserol.regression import (
    regression,
    get_labels_zohar,
    plot_roc,
    plot_confusion_matrix,
)


def makeFigure():
    axes, fig = getSetup((10, 4), (1, 2))
    # this needs to be recreated after lrank was abandoned
    return fig
    cube = prepare_data(zohar())
    x_opt_lrank, _ = optimize_loss(cube, lrank=True)
    sample, ag = reshape_params(x_opt_lrank, cube, lrank=True)
    labels, label_encoder = get_labels_zohar()
    y_pred, model, x, y = regression(sample, labels, scale_x=0, l1_ratio=0)
    ax = plot_roc(x, y, model, label_encoder, axes[0])
    ax.set(title="ROC Curves for Multi-Class Prediction (OVR)")
    ax = plot_confusion_matrix(x, y, model, label_encoder, axes[1])
    ax.set(title="Confusion Matrix for Multi-Class Prediction")
    fig.subplots_adjust(bottom=0.1)
    fig.text(
        0, 0, f"Balanced accuracy score {round(balanced_accuracy_score(y, y_pred), 2)}"
    )
    fig.text(0, 1, f"Zohar Data Multi-class Regression")
    return fig
