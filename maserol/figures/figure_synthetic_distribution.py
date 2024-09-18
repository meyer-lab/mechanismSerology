import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy

from maserol.datasets import Alter, KaplonekVaccine, Zohar
from maserol.figures.common import Multiplot
from maserol.forward_backward import generate_rtot_distribution


def makeFigure():
    """
    Plot distributions of anti-antibody detection for all datasets and
    forward-backward on separate rows.

    Args:
    n_bins (int): Number of bins for the histograms
    """

    n_bins = 20
    datasets = {
        "Zohar": Zohar(),
        "KaplonekVaccine": KaplonekVaccine(),
        "Alter": Alter(),
    }
    name_mapping = {
        "Zohar": "Zohar",
        "KaplonekVaccine": "Kaplonek",
        "Alter": "Alter",
    }

    # Set up the color palette
    colors = sns.color_palette("deep").as_hex()
    antibody_color = colors[0]
    fb_color = colors[1]
    alpha = 0.6

    def filter_ligands(ligands):
        return [
            lig for lig in ligands if re.match("^igg[1-3]", lig, flags=re.IGNORECASE)
        ]

    # Count number of subplots needed for each dataset after filtering
    dataset_subplots = {
        name: len(
            filter_ligands(
                dataset.get_detection_signal(select_ligs=False).Ligand.values
            )
        )
        for name, dataset in datasets.items()
    }

    n_rows = (
        len(datasets) + 1
    )  # One row per dataset + one row for forward-backward and aggregate
    n_cols = max(
        dataset_subplots.values()
    )  # Use the maximum number of filtered ligands as column count

    plot = Multiplot((n_cols, n_rows), fig_size=(7.15, 8.3))

    all_normalized_data = []

    ax_idx = 0

    for row, (dataset_name, dataset) in enumerate(datasets.items()):
        detection_signal = dataset.get_detection_signal(select_ligs=False)
        filtered_ligands = filter_ligands(detection_signal.Ligand.values)
        detection_signal = detection_signal.sel(Ligand=filtered_ligands)
        Ig_numbers = [
            int(re.search(r"\d+", ligand).group()) for ligand in filtered_ligands
        ]

        for col, ligand in enumerate(filtered_ligands):
            ax = plot.axes[row * n_cols + Ig_numbers[col] - 1]
            ligand_data = detection_signal.sel(Ligand=ligand).values.flatten()

            # Apply log(x+1) transformation
            log_data = np.log1p(ligand_data)

            # Normalize the log-transformed data
            normalized_data = log_data / np.mean(log_data)
            all_normalized_data.extend(normalized_data)

            sns.histplot(
                log_data,
                bins=n_bins,
                stat="density",
                kde=False,
                alpha=alpha,
                color=antibody_color,
                ax=ax,
            )
            ax.set_title(f"{name_mapping[dataset_name]}: α-{ligand} IgG")
            ax.set_xlabel("log(Detection signal + 1)")
            ax.set_ylabel("")
            plot.add_subplot_label(
                row * n_cols + Ig_numbers[col] - 1, chr(ord("a") + ax_idx)
            )
            ax_idx += 1

        # Hide any unused subplots in this row
        for col in range(n_cols):
            if col + 1 not in Ig_numbers:
                plot.axes[row * n_cols + col].set_visible(False)

    # Generate and plot histogram of the forward-backward distribution on the last row
    fb_distribution = generate_rtot_distribution(10000)  # Using 10000 samples

    # Apply log(x+1) transformation to the synthetic data
    log_fb_distribution = np.log1p(fb_distribution)

    # Normalize the log-transformed synthetic data
    normalized_fb_distribution = log_fb_distribution.flatten() / np.mean(
        log_fb_distribution
    )

    ax = plot.axes[-n_cols]  # Use the first subplot of the last row
    sns.histplot(
        log_fb_distribution.flatten(),
        bins=n_bins,
        stat="probability",
        kde=False,
        alpha=alpha,
        color=fb_color,
        label="Synthetic",
        ax=ax,
    )
    ax.set_title("Synthetic distribution")
    ax.set_xlabel("log(Fc species abundance + 1)")
    ax.set_ylabel("")
    plot.add_subplot_label(n_cols * (n_rows - 1), chr(ord("a") + ax_idx))
    ax_idx += 1

    # Plot the aggregate distribution
    ax = plot.axes[-n_cols + 1]  # Use the second subplot of the last row

    # Create a DataFrame for seaborn
    df = pd.DataFrame(
        {
            "value": np.concatenate([all_normalized_data, normalized_fb_distribution]),
            "type": ["Aggregate"] * len(all_normalized_data)
            + ["Synthetic"] * len(normalized_fb_distribution),
        }
    )

    # Use seaborn to plot both distributions
    sns.histplot(
        data=df,
        x="value",
        hue="type",
        stat="probability",
        common_norm=False,
        alpha=alpha,
        bins=n_bins,
        ax=ax,
        palette=[antibody_color, fb_color],
    )
    ax.set_title("Measurements (aggregate) vs synthetic")
    ax.set_xlabel("Normalized log(signal + 1) or log(Fc species abundance + 1)")
    ax.set_ylabel("")
    plot.add_subplot_label(n_cols * (n_rows - 1) + 1, chr(ord("a") + ax_idx))
    ax_idx += 1
    # Manually specify the handles and labels for the legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=antibody_color),
        plt.Rectangle((0, 0), 1, 1, color=fb_color),
    ]
    labels = ["Measurements", "Synthetic"]
    ax.legend(handles=handles, labels=labels, title=None)

    # Calculate KL divergence
    hist_aggregate, _ = np.histogram(all_normalized_data, bins=n_bins, density=True)
    hist_synthetic, _ = np.histogram(
        normalized_fb_distribution, bins=n_bins, density=True
    )

    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    hist_aggregate = hist_aggregate + epsilon
    hist_synthetic = hist_synthetic + epsilon

    kl_divergence = entropy(hist_aggregate, hist_synthetic)

    # Add KL divergence text to the plot
    ax.text(
        0.95,
        0.05,
        f"KL Divergence:\n{kl_divergence:.4f}",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    # Hide any unused subplots in the last row
    for col in range(2, n_cols):
        plot.axes[-n_cols + col].set_visible(False)

    # Set y-axis to scientific notation with 2 significant figures
    for ax in plot.axes:
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(8)

        # Set to 2 significant figures
        @plt.FuncFormatter
        def sci_formatter(x, p):
            return f"{x:.1e}"

        ax.yaxis.set_major_formatter(sci_formatter)

    plot.fig.tight_layout(pad=0, w_pad=0.3, h_pad=0.85)

    return plot.fig
