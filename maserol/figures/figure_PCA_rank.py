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

    plot = Multiplot((2, 2), (4.5, 2.5))
    plot.fig.suptitle("Imputation performance vs PCA rank")

    sns.boxplot(data=df, x="Ligand", y="r2", hue="Rank", ax=plot.axes[0])
    plot.axes[0].set_xlabel("Detection")
    plot.axes[0].set_ylabel("$R^2$")
    plot.axes[0].set_title("$R^2$ by Detection")

    sns.boxplot(data=df, x="Ligand", y="r", hue="Rank", ax=plot.axes[1])
    plot.axes[1].set_xlabel("Detection")
    plot.axes[1].set_ylabel("$r$")
    plot.axes[1].set_title("$r$ by Detection")

    sns.boxplot(data=df, y="r", x="Rank", ax=plot.axes[3])
    plot.axes[3].set_xlabel("Rank")
    plot.axes[3].set_ylabel("$r$")
    plot.axes[3].set_title("$r$ Total")

    sns.boxplot(data=df, y="r2", x="Rank", ax=plot.axes[2])
    plot.axes[2].set_xlabel("Rank")
    plot.axes[2].set_ylabel("$R^2$")
    plot.axes[2].set_title("$R^2$ Total")

    return plot.fig
