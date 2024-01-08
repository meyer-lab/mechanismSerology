import functools
import numpy as np
import pandas as pd
import seaborn as sns

from maserol.impute import (
    impute_missing_ms,
    run_repeated_imputation,
)
from maserol.preprocess import assemble_options, get_kaplonek_mgh_data
from maserol.figures.common import getSetup


LIGS = ["FcR2A", "FcR2B", "FcR3A", "FcR3B"]
RUNS = 3
MISSINGNESS = 0.2


def makeFigure():
    tensor = get_kaplonek_mgh_data()
    tensor = tensor[np.random.choice(tensor.shape[0], 300)]

    fs = [1, 4]
    optss = [
        assemble_options(tensor, ["IgG1", "IgG1f", "IgG3", "IgG3f"], FcR_f=f)
        for f in fs
    ]
    for opts in optss:
        opts["tol"] = 1e-6
    imputers = [functools.partial(impute_missing_ms, opts=opts) for opts in optss]
    df = pd.concat(
        run_repeated_imputation(
            tensor,
            imputer,
            ligs=LIGS,
            runs=RUNS,
            missingness=MISSINGNESS,
            imputer_name=f,
        )
        for imputer, f in zip(imputers, fs)
    )
    axes, fig = getSetup((14, 6), (1, 2))
    sns.boxplot(data=df, x="Ligand", y="r2", hue="Method", ax=axes[0])
    axes[0].set_xlabel("Ligand")
    axes[0].set_ylabel("Coefficient of Determination")
    axes[0].legend(title="Valency")
    sns.boxplot(data=df, x="Ligand", y="r", hue="Method", ax=axes[1])
    axes[1].set_xlabel("Ligand")
    axes[1].set_ylabel("Pearson Correlation")
    axes[1].legend(title="Valency")
    return fig
