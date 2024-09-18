import matplotlib.pyplot as plt
import pandas as pd

from maserol.figures.common import CACHE_DIR
from maserol.figures.figure_4 import ALTER_RTOT_CACHE_PATH
from maserol.figures.figure_5 import (
    KAPLONEK_VACCINE_RTOT_CACHE_PATH,
    ZOHAR_RTOT_CACHE_PATH,
)
from maserol.forward_backward import generate_rtot_distribution


def load_and_plot_distributions(antibody_path, n_samples=1000, n_bins=200):
    """
    Load antibody abundances from a specified path, plot their distribution
    separated by antibody type, and plot the forward-backward antibody distribution.

    Args:
    antibody_path (str): Path to the CSV file containing antibody abundances.
    n_samples (int): Number of samples to generate for the forward-backward distribution.
    """
    # Load antibody abundances
    antibody_df = (
        pd.read_csv(antibody_path)
        .reset_index(drop=True)
        .set_index(["Sample", "Antigen"])
    )

    # Plot histograms of the input dataset for each antibody type
    antibodies = antibody_df.columns
    fig, axes = plt.subplots(2, len(antibodies), figsize=(5 * len(antibodies), 10))

    for i, antibody in enumerate(antibodies):
        ax = axes[0, i]
        ax.hist(
            antibody_df[antibody],
            bins=n_bins,
            density=True,
            alpha=0.7,
        )
        ax.set_title(f"{antibody} Distribution")
        ax.set_xlabel("Abundance")
        ax.set_ylabel("Density")

    # Generate and plot histograms of the forward-backward distribution for each antibody species
    fb_distribution = generate_rtot_distribution(n_samples)

    ax = axes[1, 0]  # Use only the first subplot in the second row
    ax.hist(fb_distribution.flatten(), bins=n_bins, density=True, alpha=0.7)
    ax.set_title("Forward-Backward Distribution")
    ax.set_xlabel("Abundance")
    ax.set_ylabel("Density")

    # Hide the unused subplots in the second row
    for i in range(1, len(antibodies)):
        axes[1, i].set_visible(False)

    plt.tight_layout()
    plt.show()


# Example usage:
# load_and_plot_distributions(ZOHAR_RTOT_CACHE_PATH)
# load_and_plot_distributions(KAPLONEK_VACCINE_RTOT_CACHE_PATH)
load_and_plot_distributions(ALTER_RTOT_CACHE_PATH)
