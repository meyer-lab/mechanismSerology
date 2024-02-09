import numpy as np
import seaborn as sns

from maserol.util import assemble_Ka, DEFAULT_RCPS


def plot_Ka(cube, rcps=DEFAULT_RCPS, Ka=None, ax=None):
    Ka = Ka if Ka is not None else assemble_Ka(cube, rcps)
    ax = sns.heatmap(np.log10(Ka), ax=ax, annot=Ka, fmt=".2e")
    ax.set_title("log10 Ka")
    ax.set_xticklabels(list(rcps))
    ax.set_yticklabels(cube.Ligand.values, rotation=0)
    return ax
