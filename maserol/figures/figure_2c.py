import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import r2_score

from maserol.figures.common import getSetup
from maserol.forward_backward import forward_backward


N_ITER = 1
STEPS = 3


def makeFigure():
    axes, fig = getSetup((3.5, 2.7), (1, 1))

    noises = np.linspace(0, 0.35, STEPS)

    dfs = []

    for noise in noises:
        Rtot_pairs = [forward_backward(noise_std=noise) for _ in range(N_ITER)]
        dfs.append(
            pd.DataFrame(
                {
                    "noise": np.full(N_ITER, noise),
                    "r2": [
                        r2_score(
                            np.log10(Rtot.values.flatten()),
                            np.log10(Rtot_inferred.flatten()),
                        )
                        for Rtot, Rtot_inferred in Rtot_pairs
                    ],
                }
            )
        )

    df = pd.concat(dfs).reset_index()

    ax = sns.lineplot(data=df.reset_index(drop=True), x="noise", y="r2", ax=axes[0])
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("$r^2$")
    return fig
