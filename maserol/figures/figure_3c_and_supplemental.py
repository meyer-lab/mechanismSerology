import functools
import numpy as np
import pandas as pd
import seaborn as sns
from tensordata.zohar import data as zohar

from maserol.impute import (
    impute_missing_ms,
    impute_missing_pca,
    run_repeated_imputation,
)
from maserol.preprocess import prepare_data
from maserol.figures.common import getSetup


LIGS = ["IgG1", "IgG3", "FcR2A", "FcR2B", "FcR3A", "FcR3B"]
RUNS_PER_LIG = 3


def makeFigure():
    tensor_3d_zohar = (
        zohar()
        .rename({"Receptor": "Ligand"})
        .sel(Ligand=["IgG1", "IgG3", "FcR2A", "FcR2B", "FcR3A", "FcR3B"])
        .transpose("Sample", "Ligand", "Antigen")
    )
    tensor_3d_zohar_sub = tensor_3d_zohar[
        np.random.choice(tensor_3d_zohar.shape[0], 300)
    ]
    tensor_2d_zohar_sub = prepare_data(
        tensor_3d_zohar_sub.rename({"Ligand": "Receptor"})
    )
    imputers = [
        impute_missing_ms,
        functools.partial(impute_missing_pca, ncomp=1),
    ]
    tensors = [tensor_2d_zohar_sub, tensor_2d_zohar_sub]
    df = pd.concat(
        run_repeated_imputation(tensor, imputer, ligs=LIGS, runs=RUNS_PER_LIG)
        for tensor, imputer in zip(tensors, imputers)
    )

    df = df.replace({"impute_missing_pca": "PCA", "impute_missing_ms": "Binding model"})

    axes, fig = getSetup((6, 8), (1, 1))

    sns.boxplot(data=df, x="Ligand", y="r", hue="Method", ax=axes[0])
    axes[0].set_xlabel("Ligand")
    axes[0].set_ylabel("Pearson Correlation")

    sns.boxplot(data=df, x="Ligand", y="r2", hue="Method", ax=axes[1])
    axes[1].set_xlabel("Ligand")
    axes[1].set_ylabel("Coefficient of Determination")

    return fig