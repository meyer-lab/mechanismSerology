import functools
import numpy as np
import pandas as pd
import seaborn as sns
from tensordata.zohar import data as zohar

from maserol.impute import (
    impute_missing_pca,
    run_repeated_imputation,
)
from maserol.preprocess import prepare_data
from maserol.figures.common import getSetup, CACHE_DIR

LIGS = ["IgG1", "IgG3", "FcR2A", "FcR2B", "FcR3A", "FcR3B"]
N_CPLX = 500
MISSINGNESS = 0.1
RUNS = 5
# each row and column must have at least RANK nonmissing values
RANKS = range(1, len(LIGS))
UPDATE_CACHE = True


def makeFigure():
    file_name = "fig_S2.csv"
    if UPDATE_CACHE:
        data = prepare_data(zohar())
        data = data.sel(Ligand=["IgG1", "IgG3", "FcR2A", "FcR2B", "FcR3A", "FcR3B"])
        data = data[np.random.choice(data.shape[0], N_CPLX)]

        imputers = [
            functools.partial(impute_missing_pca, ncomp=ncomp) for ncomp in RANKS
        ]
        df = pd.concat(
            run_repeated_imputation(
                data, imputer, runs=RUNS, missingness=MISSINGNESS
            ).replace("impute_missing_pca", rank)
            for imputer, rank in zip(imputers, RANKS)
        )
        df.rename(columns={"Method": "Rank"}, inplace=True)
        df.to_csv(CACHE_DIR / file_name)
    else:
        df = pd.read_csv(CACHE_DIR / file_name)

    axes, fig = getSetup((12, 12), (2, 1))
    fig.suptitle("Imputation performance vs PCA rank")

    sns.boxplot(data=df, x="Ligand", y="r2", hue="Rank", ax=axes[0])
    axes[0].set_xlabel("Ligand")
    axes[0].set_ylabel("$r^2$")

    sns.boxplot(data=df, x="Ligand", y="r", hue="Rank", ax=axes[1])
    axes[1].set_xlabel("Ligand")
    axes[1].set_ylabel("Pearson correlation")

    return fig
