import seaborn as sns
from sklearn.metrics import roc_auc_score

from tensordata.zohar import data as zohar

from maserol.core import optimizeLoss, reshapeParams
from maserol.figures.common import getSetup
from maserol.preprocess import HIgGFs, HIgGs, prepare_data
from maserol.regression import regression, get_labels_zohar, plot_roc, plot_confusion_matrix, plot_regression_weights, add_auc_label

def makeFigure():
    cube = prepare_data(zohar())

    ab_types = HIgGFs
    x_opt_lrank, _ = optimizeLoss(cube, lrank=True, ab_types=ab_types)
    sample, ag = reshapeParams(x_opt_lrank, cube, lrank=True, ab_types=ab_types)

    ab_types = HIgGs
    x_opt_lrank_no_fucose, _ = optimizeLoss(cube, lrank=True, ab_types=ab_types)
    sample_no_fucose, ag_no_fucose = reshapeParams(x_opt_lrank_no_fucose, cube, lrank=True, ab_types=ab_types)

    labels, label_encoder = get_labels_zohar(multiclass=False)
    y_pred, model, x, y = regression(sample, labels, scale_x=0, l1_ratio=0)
    y_pred_no_fucose, model_no_fucose, x_no_fucose, y = regression(sample_no_fucose, labels, scale_x=0, l1_ratio=0)
    axes, fig = getSetup((15, 8), (2, 3))
    palette = sns.color_palette("bright", 2)
    ax = plot_roc(x, y, model, label_encoder, axes[0], label="Fucose", palette=palette[0:1], auc_label=False)
    ax.set(title="ROC Curve for Progressor vs Controller Prediction")
    ax = plot_roc(x_no_fucose, y, model_no_fucose, label_encoder, axes[0], label="No Fucose", palette=palette[1:2], auc_label=False)
    scores = [roc_auc_score(y, y_pred), roc_auc_score(y, y_pred_no_fucose)]
    add_auc_label(scores, ["Fucose", "No Fucose"], ax)

    ax = plot_confusion_matrix(x, y, model, label_encoder, axes[1])
    ax.set(title="With Fucosylated Abs")

    ax = plot_confusion_matrix(x_no_fucose, y, model_no_fucose, label_encoder, axes[2])
    ax.set(title="No Fucosylated Abs")

    axes[3].axis("off")

    ax = plot_regression_weights(model, HIgGFs, axes[4])
    ax.set(title="With Fucosylated Abs")

    ax = plot_regression_weights(model_no_fucose, HIgGs, axes[5])
    ax.set(title="No Fucosylated Abs")

    return fig