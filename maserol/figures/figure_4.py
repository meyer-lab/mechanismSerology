import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from maserol.core import optimize_loss
from maserol.datasets import Alter
from maserol.figures.common import Multiplot
from maserol.util import Rtot_to_df, assemble_options, compute_fucose_ratio


def makeFigure():
    plot = Multiplot((4, 4), (2, 1))
    figure_4b(plot.axes[1])
    plot.add_subplot_labels()
    plot.fig.tight_layout()
    return plot.fig


def figure_4b(ax):
    detection_signal = Alter().get_detection_signal()
    detection_signal = detection_signal.sel(Complex=pd.IndexSlice[:, "gp120.SF162"])
    fucose_ce = Alter().get_fucose_data()
    rcps = ["IgG1", "IgG1f", "IgG3", "IgG3f"]
    opts = assemble_options(detection_signal, rcps=rcps)
    params, _ = optimize_loss(detection_signal, **opts, return_reshaped_params=True)
    fucose_inferred = compute_fucose_ratio(
        Rtot_to_df(params["Rtot"], data=detection_signal, rcps=rcps).xs(
            "gp120.SF162", level="Antigen"
        )
    )
    fucose_compare = pd.merge(
        left=fucose_inferred, right=fucose_ce, on="Sample", how="inner"
    )

    sns.scatterplot(
        data=fucose_compare, x="fucose_ce", y="fucose_inferred", ax=ax, alpha=0.8
    )
    ax.set_xlabel("Measured IgG Fucosylation (%)")
    ax.set_ylabel("Inferred IgG Fucosylation (%)")
    r, p = pearsonr(fucose_compare["fucose_ce"], fucose_compare["fucose_inferred"])
    ax.set_title("Model Inferences vs CE Measurements")
    ax.text(
        0.75,
        0.05,
        r"r=" + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.text(
        0.75,
        0.01,
        r"p=" + "{:.2e}".format(p),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
