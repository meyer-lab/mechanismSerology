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


LIGS = ["IgG1", "IgG3", "FcR2A", "FcR2B", "FcR3A", "FcR3B"]
N_CPLX = 300
RUNS = 3
MISSINGNESS = (0.1, 0.99)

# each row and column must have at least RANK nonmissing values 
RANKS = range(1, 3)


def makeFigure():
    data = prepare_data(zohar())
    data = data.sel(Ligand=["IgG1", "IgG3", "FcR2A", "FcR2B", "FcR3A", "FcR3B"])
    data = data[
        np.random.choice(data.shape[0], N_CPLX)
    ]

    imputers = [
        functools.partial(impute_missing_pca, ncomp=ncomp) for ncomp in RANKS
    ]

    axes, fig = getSetup((10, 5 * len(MISSINGNESS)), (len(MISSINGNESS), 1))

    for i, missingness in enumerate(MISSINGNESS):
        df = pd.concat(
            run_repeated_imputation(
                data, imputer, runs=RUNS, missingness=missingness
            ).replace("impute_missing_pca", rank)
            for imputer, rank in zip(imputers, RANKS)
        )

        df.rename(columns={"Method": "Rank"}, inplace=True)

        sns.boxplot(data=df, x="Ligand", y="r", hue="Rank", ax=axes[i])
        axes[i].set_xlabel("Ligand")
        axes[i].set_ylabel("Pearson Correlation")
        axes[i].set_title(f"Missingness: {missingness}")
    return fig
