import functools
import numpy as np
import pandas as pd
import seaborn as sns

from maserol.datasets import Zohar
from maserol.impute import (
    impute_missing_pca,
    run_repeated_imputation,
)
from maserol.figures.common import CACHE_DIR, Multiplot

N_CPLX = 500
MISSINGNESS = 0.1
RUNS = 5
UPDATE_CACHE = False


def makeFigure():
    file_name = "fig_S2.csv"
    if UPDATE_CACHE:
        data = Zohar().get_detection_signal()
        # each row and column must have at least RANK nonmissing values
        ranks = range(1, len(data.Ligand.values))
        data = data[np.random.choice(data.shape[0], N_CPLX)]

        imputers = [
            functools.partial(impute_missing_pca, ncomp=ncomp) for ncomp in ranks
        ]
        df = pd.concat(
            run_repeated_imputation(
                data, imputer, runs=RUNS, missingness=MISSINGNESS
            ).replace("impute_missing_pca", rank)
            for imputer, rank in zip(imputers, ranks)
        )
        df.rename(columns={"Method": "Rank"}, inplace=True)
        df.to_csv(CACHE_DIR / file_name)
    else:
        df = pd.read_csv(CACHE_DIR / file_name)

    plot = Multiplot((6, 3), (2, 1))
    plot.fig.suptitle("Imputation performance vs PCA rank")

    sns.boxplot(data=df, x="Ligand", y="r2", hue="Rank", ax=plot.axes[0])
    plot.axes[0].set_xlabel("Ligand")
    plot.axes[0].set_ylabel("$r^2$")

    sns.boxplot(data=df, x="Ligand", y="r", hue="Rank", ax=plot.axes[1])
    plot.axes[1].set_xlabel("Ligand")
    plot.axes[1].set_ylabel("Pearson correlation")

    return plot.fig
