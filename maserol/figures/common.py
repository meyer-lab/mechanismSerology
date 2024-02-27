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

DETECTION_DISPLAY_NAMES = {
    "IgG1": "α-HIgG1",
    "IgG3": "α-HIgG3",
    "FcR2A": "FcγRIIA",
    "FcR2B": "FcγRIIB",
    "FcR3A": "FcγRIIIA",
    "FcR3B": "FcγRIIIB",
}

PLOT_CONFIG = dict(
    style="whitegrid",
    font_scale=0.7,
    color_codes=True,
    palette="colorblind",
    rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
)


class Multiplot:
    def __init__(
        self, grid, ax_size=None, fig_size=None, subplot_specs=None, empty=None
    ):
        """
        :param ax_size: Tuple specifying the axis size (width, height)
        :param grid: Tuple specifying the grid size (columns, rows)
        :param subplot_specs: List of tuples specifying subplot positions and spans as
                              (col_start, col_span, row_start, row_span) for each subplot
        """
        if ax_size is None:
            self.fig_size = fig_size
            self.ax_size = (fig_size[0] / grid[0], fig_size[1] / grid[1])
        elif fig_size is None:
            self.ax_size = ax_size
            self.fig_size = (ax_size[0] * grid[0], ax_size[1] * grid[1])
        else:
            raise ValueError("Must pass in either ax_size or fig_size")
        self.grid = grid
        if subplot_specs is None:
            subplot_specs = [
                (i, 1, j, 1) for j in range(self.grid[1]) for i in range(self.grid[0])
            ]
        self.subplot_specs = subplot_specs
        self.empty = empty if empty is not None else set()
        self.axes, self.fig = self.setup()

    def setup(self):
        """Establish figure setup with flexible subplots."""
        sns.set(**PLOT_CONFIG)

        # Setup figure and grid
        f = plt.figure(figsize=self.fig_size, constrained_layout=True)
        gs = gridspec.GridSpec(self.grid[1], self.grid[0], figure=f)

        # Create subplots according to the specifications
        ax = []
        for spec in self.subplot_specs:
            col_start, col_span, row_start, row_span = spec
            ax.append(
                f.add_subplot(
                    gs[
                        row_start : row_start + row_span,
                        col_start : col_start + col_span,
                    ]
                )
            )

        for ax_i in self.empty:
            ax[ax_i].axis("off")

        return ax, f

    def add_subplot_labels(self, labels=None):
        """Add labels to subplots. If labels are None, use alphabetical labels."""
        if labels is None:
            labels = [chr(ord("a") + i) for i in range(len(self.axes))]
        for ax, label in zip(self.axes, labels):
            ax.set_title(label, loc="left", fontsize=16, fontweight="bold")

    def subplot_spec_to_bounds(self, spec):
        x_width = 1 / self.grid[0]
        y_height = 1 / self.grid[1]
        x_left = spec[0] * x_width
        width = spec[1] * x_width
        y_bot = (self.grid[1] - (spec[2] + spec[3])) * y_height
        height = spec[3] * y_height
        return (x_left, y_bot, width, height)

    def add_subplot_label(self, ax_index, label, ax_relative=False):
        # this is just dealing with matplotlib stupidity. offer both choices and
        # the user can pick, but either might break.
        if ax_relative:
            left = 0.2 / self.subplot_specs[ax_index][1]
            up = 0.1 / self.subplot_specs[ax_index][3]
            self.axes[ax_index].text(
                0 - left,
                1 + up,
                label,
                transform=self.axes[ax_index].transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
            )
        else:
            bounds = self.subplot_spec_to_bounds(self.subplot_specs[ax_index])
            self.fig.text(
                bounds[0],
                bounds[1] + 1.02 * bounds[3],
                label,
                va="top",
                ha="left",
                fontsize=16,
                fontweight="bold",
            )

    def add_subplot_labels(self, ax_relative=False):
        for i in range(len(self.subplot_specs)):
            self.add_subplot_label(i, chr(ord("a") + i), ax_relative=ax_relative)


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


def annotate_mann_whitney(annotator: Annotator, correction="Benjamini-Hochberg"):
    """Perform Mann-Whitney test and multiple hypothesis correction and annotate."""
    annotator.configure(
        test="Mann-Whitney",
        text_format="star",
        comparisons_correction=correction,
        correction_format="replace",
        loc="outside",
    )
    annotator.apply_test()
    remove_ns_annotations(annotator)
    annotator.annotate()
