import numpy as np
import seaborn as sns

from maserol.util import DEFAULT_RCPS, assemble_Ka


def plot_Ka(ligs, rcps=DEFAULT_RCPS, Ka=None, ax=None):
    Ka = Ka if Ka is not None else assemble_Ka(ligs, rcps)
    ax = sns.heatmap(np.log10(Ka), ax=ax, annot=Ka, fmt=".2e")
    ax.set_title("log10 Ka")
    ax.set_xticklabels(list(rcps))
    ax.set_yticklabels(ligs, rotation=0)
    return ax
