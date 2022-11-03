from sklearn.metrics import balanced_accuracy_score

from tensordata.zohar import data3D as zohar

from maserol.core import optimizeLoss, reshapeParams
from maserol.figures.common import getSetup
from maserol.preprocess import HIgGFs, prepare_data
from maserol.regression import regression, get_labels_zohar, plot_roc, plot_confusion_matrix, plot_regression_weights

def makeFigure():
    ab_types = HIgGFs
    cube = prepare_data(zohar(xarray=True, logscale=False))
    x_opt_lrank, _ = optimizeLoss(cube, lrank=True, ab_types=ab_types)
    sample, ag = reshapeParams(x_opt_lrank, cube, lrank=True, ab_types=ab_types)
    labels, label_encoder = get_labels_zohar(multiclass=False)
    y_pred, model, x, y = regression(sample, labels, scale_x=0, l1_ratio=0)
    axes, fig = getSetup((15, 4), (1, 3))
    ax = plot_roc(x, y, model, label_encoder, axes[0])
    ax.set(title="ROC Curve for Progressor vs Controller Prediction")
    ax = plot_confusion_matrix(x, y, model, label_encoder, axes[1])
    ax.set(title="Confusion Matrix for Progressor vs Controller Prediction")
    ax = plot_regression_weights(model, ab_types, axes[2])
    ax.set(title="Regression Weights for Progressor vs Controller Prediciton")
    fig.text(0, 0, f"Balanced accuracy score {round(balanced_accuracy_score(y, y_pred), 2)}")
    fig.text(0, 1, f"Zohar Data Controller vs. Progressor Regression")
    return fig
