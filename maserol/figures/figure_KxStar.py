from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

from maserol.figures.common import CACHE_DIR, Multiplot
from maserol.forward_backward import forward_backward
from maserol.util import HIgGs

UPDATE_CACHE = False
CACHE_PATH = Path(CACHE_DIR, "KxStar_perturb.csv")


def convert_to_df(Rtot):
    Rtot_df = (
        Rtot.to_dataframe(name="Abundance")
        .reset_index(level="Receptor")
        .pivot(columns="Receptor")
    )
    Rtot_df.columns = [col[1] for col in Rtot_df.columns]
    Rtot_df.reset_index(inplace=True)
    return Rtot_df


def combine_actual_inferred(Rtot_df, Rtot_inferred):
    Rtot_inferred["Complex"] = Rtot_df["Complex"]

    Rtot_inferred_melt = pd.melt(
        Rtot_inferred,
        id_vars=["Perturbation", "Complex"],
        var_name="Receptor",
        value_name="Inferred abundance",
    )
    Rtot_melt = pd.melt(
        Rtot_df, id_vars=["Complex"], var_name="Receptor", value_name="Actual abundance"
    )

    Rtot_combined = pd.merge(Rtot_inferred_melt, Rtot_melt, on=["Complex", "Receptor"])

    return Rtot_combined


def makeFigure():
    rcps = HIgGs
    # This is wonky because we originally collected this data over multiple runs
    KxStar_perturb = np.logspace(-2, 2, 10)
    interval = 4 / 9
    up = np.arange(2, 3.5, interval)[1:]
    down = up * -1
    KxStar_perturb = np.concatenate([KxStar_perturb, 10 ** np.concatenate([down, up])])
    n_cplx_per_perturb = 1000

    if UPDATE_CACHE:
        Rtot, Rtot_inferred = forward_backward(
            KxStar_perturb, n_cplx=n_cplx_per_perturb * len(KxStar_perturb), rcps=rcps
        )
        combined = combine_actual_inferred(convert_to_df(Rtot), Rtot_inferred)
        combined.to_csv(CACHE_PATH, index=False)
    else:
        combined = pd.read_csv(CACHE_PATH)

    plot = Multiplot(
        (1, 1),
        fig_size=(5, 4),
    )

    # plot the R2 for each receptor and perturbation (perturbation as x and
    # receptor as hue)
    r2_scores = []

    combined["Perturbation"] = np.log10(combined["Perturbation"])

    for perturbation in np.unique(combined["Perturbation"]):
        for rcp in rcps:
            combined_rcp = combined[
                (combined["Receptor"] == rcp)
                & (combined["Perturbation"] == perturbation)
            ]
            r2 = r2_score(
                combined_rcp["Actual abundance"], combined_rcp["Inferred abundance"]
            )
            r2_scores.append({"Perturbation": perturbation, "Receptor": rcp, "R2": r2})

    r2_df = pd.DataFrame(r2_scores)

    ax = sns.lineplot(
        data=r2_df,
        x="Perturbation",
        y="R2",
        hue="Receptor",
        marker="o",
        ax=plot.axes[0],
    )
    ax.set_title("Prediction performance vs Kx* perturbation")
    ax.set_xlabel("Kx* Perturbation coefficient")
    ax.set_ylabel("$R^2$")
    ax.legend(title="Fc species")
    # set x tick labels as 10^x
    ax.set_xticklabels([f"$10^{{{round(tick)}}}$" for tick in ax.get_xticks()])

    plot.fig.tight_layout()

    return plot.fig
