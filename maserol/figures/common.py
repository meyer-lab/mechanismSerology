"""
This file contains functions that are used in multiple figures.
"""

import sys
import time
import logging
from pathlib import Path
from string import ascii_lowercase

import matplotlib
import seaborn as sns
from matplotlib import gridspec, pyplot as plt
from statannotations.Annotator import Annotator


matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35

THIS_DIR = Path(__file__).parent
CACHE_DIR = THIS_DIR.parent / "data" / "cache"


def getSetup(figsize, gridd, multz=None, empts=None):
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = dict()

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = list()
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def add_subplot_label(ax, label):
    ax.text(
        -0.2,
        1.2,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )


def add_subplot_labels(axs):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        add_subplot_label(ax, ascii_lowercase[ii])


def genFigure():
    """Main figure generation function."""
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec("from maserol.figures." + nameOut + " import makeFigure", globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=300, bbox_inches="tight", pad_inches=0)

    logging.info(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.")


def remove_ns_annotations(annotator: Annotator):
    """The provided annotator has no way to exclude annotating pairs which are
    not significant. This prevents those ns annotations from showing."""
    annotator.annotations = [an for an in annotator.annotations if an.text != "ns"]


def annotate_mann_whitney(annotator: Annotator):
    """Perform Mann-Whitney test and multiple hypothesis correction and annotate."""
    annotator.configure(
        test="Mann-Whitney", text_format="star", comparisons_correction="Bonferroni"
    )
    annotator.apply_test()
    remove_ns_annotations(annotator)
    annotator.annotate()
