import functools
import numpy as np
import pandas as pd
import seaborn as sns
from tensordata.zohar import data as zohar

from maserol.impute import (
    impute_missing_cp,
    impute_missing_ms,
    impute_missing_pca,
    run_repeated_imputation,
)
from maserol.preprocess import prepare_data
from maserol.figures.common import getSetup


# MS 100% missing ligand vs CP and PCA 95% missing
LIGS = ["FcR2A", "FcR2B", "FcR3A", "FcR3B"]


def makeFigure():
    tensor_3d_zohar = (
        zohar()
        .rename({"Receptor": "Ligand"})
        .sel(Ligand=["IgG1", "IgG3", "FcR2A", "FcR2B", "FcR3A", "FcR3B"])
        .transpose("Sample", "Ligand", "Antigen")
    )
    tensor_3d_zohar_sub = tensor_3d_zohar[
        np.random.choice(tensor_3d_zohar.shape[0], 200)
    ]
    tensor_2d_zohar_sub = prepare_data(
        tensor_3d_zohar_sub.rename({"Ligand": "Receptor"})
    )
    imputers = [
        impute_missing_ms,
        functools.partial(impute_missing_cp, ncomp=5),
        functools.partial(impute_missing_pca, ncomp=5),
    ]
    tensors = [tensor_2d_zohar_sub, tensor_3d_zohar_sub, tensor_2d_zohar_sub]
    missingnesss = [1, 0.95, 0.95]
    runss = [1, 3, 3]
    df = pd.concat(
        run_repeated_imputation(
            tensor, imputer, ligs=LIGS, runs=runs, missingness=missingness
        )
        for tensor, imputer, missingness, runs in zip(
            tensors, imputers, missingnesss, runss
        )
    )
    axes, fig = getSetup((7, 6), (1, 1))
    sns.boxplot(data=df, x="Ligand", y="r", hue="Method", ax=axes[0])
    axes[0].set_xlabel("Ligand")
    axes[0].set_ylabel("Pearson Correlation")
    return fig
