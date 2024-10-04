import functools

import numpy as np
import pandas as pd
import seaborn as sns

from maserol.datasets import Zohar
from maserol.figures.common import CACHE_DIR, DETECTION_DISPLAY_NAMES, Multiplot
from maserol.impute import (
    impute_missing_pca,
    run_repeated_imputation,
)

N_CPLX = 500
MISSINGNESS = 0.1
RUNS = 5
UPDATE_CACHE = False
TITLE_FONT_SIZE = 8


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
            for imputer, rank in zip(imputers, ranks, strict=False)
        )
        df.rename(columns={"Method": "Rank"}, inplace=True)
        df.to_csv(CACHE_DIR / file_name)
    else:
        df = pd.read_csv(CACHE_DIR / file_name)

    plot = Multiplot(
        (3, 2),
        fig_size=(6, 4),
        subplot_specs=[
            (0, 2, 0, 1),
            (2, 1, 0, 1),
            (0, 2, 1, 1),
            (2, 1, 1, 1),
        ],
    )

    ax = plot.axes[0]
    sns.barplot(data=df, x="Rank", y="r2", hue="Ligand", ax=ax)
    ax.set_xlabel(None)
    ax.set_ylabel("Imputation accuracy ($R^2$)", fontsize=TITLE_FONT_SIZE)
    ax.set_title("Accuracy by detection", fontsize=TITLE_FONT_SIZE)
    ax.set_xticklabels([])
    handles, labels = ax.get_legend_handles_labels()
    labels = [DETECTION_DISPLAY_NAMES[label] for label in labels]
    ax.legend(handles, labels, loc="lower left").get_frame().set_alpha(1)

    ax = plot.axes[1]
    sns.barplot(data=df, y="r2", x="Rank", ax=ax)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticklabels([])
    ax.set_title("Total accuracy", fontsize=TITLE_FONT_SIZE)

    ax = plot.axes[2]
    sns.barplot(data=df, x="Rank", y="r", hue="Ligand", ax=ax)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Imputation accuracy ($r$)", fontsize=TITLE_FONT_SIZE)
    ax.legend().remove()
    # handles, labels = ax.get_legend_handles_labels()
    # labels = [DETECTION_DISPLAY_NAMES[label] for label in labels]
    # ax.legend(handles, labels, loc="lower left").get_frame().set_alpha(1)

    ax = plot.axes[3]
    sns.barplot(data=df, y="r", x="Rank", ax=ax)
    ax.set_xlabel("Rank")
    ax.set_ylabel(None)

    plot.add_subplot_labels()
    # plot.fig.tight_layout(pad=0, w_pad=0.2, h_pad=2)

    return plot.fig
