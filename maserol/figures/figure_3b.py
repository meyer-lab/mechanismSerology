from tensordata.zohar import data as zohar

from maserol.figures.common import getSetup
from maserol.impute import (
    assemble_residual_mask,
    impute_missing_ms,
    imputation_scatterplot,
)
from maserol.preprocess import prepare_data


def makeFigure():
    data = prepare_data(zohar())
    data = data.sel(Ligand=[l for l in data.Ligand.values if l != "IgG2"])
    missingness = {"FcR3B": 0.1}
    residual_mask = assemble_residual_mask(data, missingness)
    Lbound = impute_missing_ms(data, residual_mask)
    axes, fig = getSetup((3.5, 2.7), (1, 1))
    imputation_scatterplot(data, Lbound, residual_mask, axes[0])
    axes[0].set_xlabel(r"$\mathrm{log_{10}}$ Predicted FcγR3A")
    axes[0].set_ylabel(r"$\mathrm{log_{10}}$ Measured FcγR3A")
    return fig
