from .common import getSetup, subplotLabel
from ..core import *
from tensordata.atyeo import data as atyeo
from scipy.stats import pearsonr

def makeFigure():
    axs, f = getSetup((8, 4), (1, 2))

    d = atyeo()
    cube = d.tensor
    _, rec_names, ant_names = d.axes
    RKa_opt = optimize_lossfunc(cube, n_ab=1, maxiter=1000)

    # heatmap for correlation
    plot_correlation_heatmap(axs[0], RKa_opt, cube, rec_names, ant_names)

    subplotLabel(axs)
    return f


def plot_correlation_heatmap(ax, RKa_opt, cube, rec_names, ant_names):
    """
    Uses optimal parameters from optimize_lossfunc to run the model
    Generates prelim figures to compare experimental and model results
    R_subj, R_Ag, Ka, L0=L0, KxStar=KxStar
    """

    R_subj, R_Ag, Ka = reshapeParams(RKa_opt, cube)
    Lbound_model = infer_Lbound(R_subj, R_Ag, Ka, L0=1e-9, KxStar=1e-12)

    coeff = np.zeros([cube.shape[1], cube.shape[2]])
    for ii in range(cube.shape[1]):
        for jj in range(cube.shape[2]):
            coeff[ii, jj], _ = pearsonr(cube[:, ii, jj], Lbound_model[:, ii, jj])

    print(coeff)
    ax.imshow(coeff)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(ant_names)))
    ax.set_xticklabels(ant_names, rotation=45)
    ax.set_yticks(np.arange(len(rec_names)))
    ax.set_yticklabels(rec_names)

    # Loop over data dimensions and create text annotations.
    for i in range(len(ant_names)):
        for j in range(len(rec_names)):
            text = ax.text(i, j, round(coeff[j, i], 2), ha="center", va="center", color="w")
