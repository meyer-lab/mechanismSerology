import numpy as np

from maserol.figures.common import getSetup
from maserol.impute import (
    assemble_residual_mask,
    impute_missing_ms,
    imputation_scatterplot,
)
from maserol.preprocess import get_kaplonek_mgh_data


def makeFigure():
    axes, fig = getSetup((3.5, 2.7), (1, 1))
    missingness = {"IgG1": 0.1}
    mgh = get_kaplonek_mgh_data()
    mgh = mgh[np.random.choice(mgh.shape[0], 1000)]

    residual_mask = assemble_residual_mask(mgh, missingness)
    Lbound = impute_missing_ms(mgh, residual_mask)
    imputation_scatterplot(mgh, Lbound, residual_mask, axes[0])

    axes[0].set_xlabel(r"$\mathrm{log_{10}}$ Predicted IgG1")
    axes[0].set_ylabel(r"$\mathrm{log_{10}}$ Measured IgG1")
    return fig
