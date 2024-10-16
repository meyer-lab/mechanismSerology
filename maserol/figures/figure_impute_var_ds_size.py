import numpy as np
import pandas as pd
import seaborn as sns

from maserol.datasets import Zohar
from maserol.figures.common import CACHE_DIR, DETECTION_DISPLAY_NAMES, Multiplot
from maserol.figures.figure_3 import R2_FIGURE_RANGE, R_FIGURE_RANGE
from maserol.impute import impute_missing_ms, run_repeated_imputation

np.random.seed(42)

UPDATE_CACHE = False


def makeFigure():
    cache_path = CACHE_DIR / "impute_var_ds_size.csv"
    ds_sizes = np.arange(0.1, 1.01, 0.1)
    if UPDATE_CACHE:
        runs = 2
        missingness = 0.5

        data = Zohar().get_detection_signal()

        dfs = []
        for ds_size in ds_sizes:
            df_iter = run_repeated_imputation(
                data[np.random.choice(data.shape[0], int(ds_size * len(data)))],
                impute_missing_ms,
                runs=runs,
                missingness=missingness,
                ligs=[lig for lig in data.Ligand.values if lig not in ["IgG1", "IgG3"]],
            )
            df_iter["ds_size"] = ds_size
            dfs.append(df_iter)
            print(f"Finished {ds_size} {len(dfs)}")
        df = pd.concat(dfs)
        df.reset_index(inplace=True, drop=True)
        df.to_csv(cache_path)
    else:
        df = pd.read_csv(cache_path)
    df = df.rename(columns={"Ligand": "Detection"})
    df["Missingness"] = df["Missingness"] * 100
    if "ds_size" not in df.columns:
        df["ds_size"] = np.repeat(ds_sizes, len(df) // len(ds_sizes))

    df["ds_size"] = df["ds_size"] * 100

    plot = Multiplot(
        (2, 1),
        fig_size=(6, 3),
    )

    ax = plot.axes[0]
    sns.lineplot(data=df, x="ds_size", y="r", hue="Detection", ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    labels = [DETECTION_DISPLAY_NAMES[label] for label in labels]
    ax.legend(handles, labels, loc="lower left").get_frame().set_alpha(1)
    ax.set_ylabel("Imputation performance ($r$)")
    ax.set_xlabel("Dataset size (%)")
    ax.set_ylim(R_FIGURE_RANGE)
    ax.set_xticks(ds_sizes * 100)
    ax.set_title("Imputation at 50% missing values")

    ax = plot.axes[1]
    sns.lineplot(data=df, x="ds_size", y="r2", hue="Detection", ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    labels = [DETECTION_DISPLAY_NAMES[label] for label in labels]
    ax.legend(handles, labels, loc="lower left").get_frame().set_alpha(1)
    ax.set_ylabel("Imputation performance ($R^2$)")
    ax.set_xlabel("Dataset size (%)")
    ax.set_ylim((0, R2_FIGURE_RANGE[1]))
    ax.set_xticks(ds_sizes * 100)
    ax.set_title("Imputation at 50% missing values")

    plot.fig.tight_layout()
    plot.add_subplot_labels(ax_relative=True)

    return plot.fig
