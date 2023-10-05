import numpy as np
import seaborn as sns

from maserol.figures.common import getSetup
from maserol.forward_backward import forward_backward


def makeFigure():
    Rtot, Rtot_inferred = forward_backward(0)
    axes, fig = getSetup((3.5, 2.7), (1, 1))
    ax = sns.scatterplot(x=np.log10(Rtot_inferred[:, 0]), y=np.log10(Rtot[:, 0]), alpha=0.6, ax=axes[0])
    ax.set_xlabel("log10 Inferred IgG1")
    ax.set_ylabel("log10 Actual IgG1")
    return fig
