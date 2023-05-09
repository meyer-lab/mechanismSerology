import numpy as np
from tensordata.atyeo import data as atyeo
from scipy.stats import pearsonr

from .common import getSetup, subplotLabel
from ..core import reshapeParams, inferLbound, optimizeLoss
from ..preprocess import prepare_data


def makeFigure():
    # this test is still a bit rundown.
    axs, f = getSetup((8, 4), (1, 2))

    data = prepare_data(atyeo(), data_id="atyeo")

    RKa_opt, _ = optimizeLoss(data)

    # heatmap for correlation
    # TODO: BROKEN
    # plot_correlation_heatmap(axs[0], RKa_opt, data)

    subplotLabel(axs)
    return f


def plot_correlation_heatmap(ax, RKa_opt, data):
    """
    Uses optimal parameters from optimize_lossfunc to run the model
    Generates prelim figures to compare experimental and model results
    R_subj, R_Ag, Ka, L0=L0, KxStar=KxStar
    """
    R_subj, R_Ag, Ka = reshapeParams(RKa_opt, data)
    Lbound_model = inferLbound(R_subj, R_Ag, Ka, L0=1e-9, KxStar=1e-12)

    coeff = np.zeros(data.shape[1:3])
    for ii in range(data.shape[1]):
        for jj in range(data.shape[2]):
            coeff[ii, jj] = pearsonr(data[:, ii, jj], Lbound_model[:, ii, jj])[0]

    print(coeff)
    ax.imshow(coeff)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.coords[1], rotation=45)
    ax.set_yticks(np.arange(data.shape[2]))
    ax.set_yticklabels(data.coords[2])

    # Loop over data dimensions and create text annotations.
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            ax.text(i, j, round(coeff[j, i], 2), ha="center", va="center", color="w")
