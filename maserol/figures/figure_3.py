import functools
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from tensordata.zohar import data as zohar


from maserol.figures.common import getSetup, add_subplot_labels
from maserol.impute import (
    assemble_residual_mask,
    imputation_scatterplot,
    impute_missing_ms,
    impute_missing_pca,
    run_repeated_imputation,
)
from maserol.preprocess import prepare_data

THIS_DIR = Path(__file__).parent
CACHE_DIR = THIS_DIR.parent / "data" / "cache"
UPDATE_CACHE = {"3b": False, "3c": False}
LIGS_3C = ["IgG1", "IgG3", "FcR2A", "FcR2B", "FcR3A", "FcR3B"]
RUNS_PER_LIG_3C = 3


def makeFigure():
    axes, fig = getSetup((10, 3), (1, 3))
    figure_3b(axes[1])
    figure_3c(axes[2])
    add_subplot_labels(axes)
    return fig


def figure_3b(ax):
    data = prepare_data(zohar())
    data = data.sel(Ligand=[l for l in data.Ligand.values if l != "IgG2"])
    missingness = {"FcR3B": 0.1}
    residual_mask = assemble_residual_mask(data, missingness)
    filename = "fig_3b_imputation.txt"
    if UPDATE_CACHE["3b"]:
        Lbound = impute_missing_ms(data, residual_mask)
        np.savetxt(CACHE_DIR / filename, Lbound, fmt="%d")
    else:
        Lbound = np.loadtxt(CACHE_DIR / filename, dtype=float)
    imputation_scatterplot(data, Lbound, residual_mask, ax)
    ax.set_xlabel(r"$\mathrm{log_{10}}$ Predicted FcγR3A")
    ax.set_ylabel(r"$\mathrm{log_{10}}$ Measured FcγR3A")


def figure_3c(ax):
    data = prepare_data(zohar())
    data = data.sel(Ligand=[l for l in data.Ligand.values if l != "IgG2"])
    data = data[:500]

    imputers = [
        impute_missing_ms,
        functools.partial(impute_missing_pca, ncomp=1),
    ]

    filename = "fig_3c_metrics.csv"
    if UPDATE_CACHE["3c"]:
        df = pd.concat(
            run_repeated_imputation(data, imputer, ligs=LIGS_3C, runs=RUNS_PER_LIG_3C)
            for imputer in imputers
        )
        df.to_csv(CACHE_DIR / filename)
    else:
        df = pd.read_csv(CACHE_DIR / filename)

    df = df.replace({"impute_missing_pca": "PCA", "impute_missing_ms": "Binding model"})

    sns.barplot(data=df, x="Ligand", y="r", hue="Method", ax=ax)
    ax.set_xlabel("Ligand")
    ax.set_ylabel("Imputation Performance (Pearson Correlation)")
    ax.legend(title=None)
