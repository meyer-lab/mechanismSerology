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


class Multiplot:
    def __init__(self, ax_size, grid, multz=None, empts=None):
        self.ax_size = ax_size
        self.grid = grid
        self.fig_size = (self.ax_size[0] * self.grid[0], self.ax_size[1] * self.grid[1])
        self.multz = multz if multz is not None else {}
        self.empts = empts if empts is not None else []
        self.axes, self.fig = self.setup()

    def setup(self):
        """Establish figure set-up with subplots."""
        sns.set(
            style="whitegrid",
            font_scale=0.7,
            color_codes=True,
            palette="colorblind",
            rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
        )

        # Setup plotting space and grid
        f = plt.figure(figsize=self.fig_size, constrained_layout=True)
        gs1 = gridspec.GridSpec(self.grid[1], self.grid[0], figure=f)

        # Get list of axis objects
        x = 0
        ax = list()
        while x < self.grid[0] * self.grid[1]:
            if x not in self.empts:
                if x in self.multz.keys():
                    ax.append(f.add_subplot(gs1[x : x + self.multz[x] + 1]))
                else:
                    ax.append(f.add_subplot(gs1[x]))
            if x in self.multz.keys():
                x += self.multz[x]
            x += 1

        return ax, f

    def add_subplot_labels(self):
        x_width = 1 / self.grid[0]
        y_height = 1 / self.grid[1]
        skip = [k + i for k in self.multz.keys() for i in range(1, self.multz[k] + 1)]
        print(skip)
        for y in range(self.grid[1]):
            for x in range(self.grid[0]):
                n = (self.grid[1] - y - 1) * self.grid[0] + x
                if n in skip:
                    continue
                ax_index = n - sum(1 for i in skip if i <= n)
                print(n)
                print(ax_index)
                self.fig.text(
                    x * x_width,
                    (y + 1) * y_height,
                    chr(ord("a") + ax_index),
                    va="top",
                    ha="left",
                    fontsize=16,
                    fontweight="bold",
                )


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
        test="Mann-Whitney",
        text_format="star",
        comparisons_correction="Bonferroni",
        loc="outside",
    )
    annotator.apply_test()
    remove_ns_annotations(annotator)
    annotator.annotate()
