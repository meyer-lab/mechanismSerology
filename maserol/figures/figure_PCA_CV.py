import functools
import numpy as np
import pandas as pd
import seaborn as sns

from maserol.datasets import Zohar
from maserol.impute import (
    impute_missing_pca,
    run_repeated_imputation,
)
from maserol.figures.common import Multiplot


N_CPLX = 300
RUNS = 3
MISSINGNESS = (0.1, 0.99)

# each row and column must have at least RANK nonmissing values
RANKS = range(1, 3)


def makeFigure():
    data = Zohar().get_detection_signal()
    data = data[np.random.choice(data.shape[0], N_CPLX)]

    imputers = [functools.partial(impute_missing_pca, ncomp=ncomp) for ncomp in RANKS]

    plot = Multiplot((8, 4), (1, len(MISSINGNESS)))

    for i, missingness in enumerate(MISSINGNESS):
        df = pd.concat(
            run_repeated_imputation(
                data, imputer, runs=RUNS, missingness=missingness
            ).replace("impute_missing_pca", rank)
            for imputer, rank in zip(imputers, RANKS)
        )

        df.rename(columns={"Method": "Rank"}, inplace=True)

        sns.boxplot(data=df, x="Ligand", y="r", hue="Rank", ax=plot.axes[i])
        plot.axes[i].set_xlabel("Ligand")
        plot.axes[i].set_ylabel("Pearson Correlation")
        plot.axes[i].set_title(f"Missingness: {missingness}")
    return plot.fig
