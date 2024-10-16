import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from statannotations.Annotator import Annotator

from maserol.core import optimize_loss
from maserol.datasets import Alter, Zohar, data_to_df
from maserol.figures.common import (
    ANNOTATION_FONT_SIZE,
    CACHE_DIR,
    DETECTION_DISPLAY_NAMES,
    LOG10_SYMBOL,
    Multiplot,
    annotate_mann_whitney,
)
from maserol.util import Rtot_to_df, assemble_options, compute_fucose_ratio

ALPHA = 0.72
UPDATE_CACHE = {
    "zohar": False,
    "alter": False,
}
ALTER_RTOT_CACHE_PATH = CACHE_DIR / "alter_Rtot.csv"


def makeFigure():
    plot = Multiplot(
        (6, 3),
        fig_size=(7.5, 8),
        subplot_specs=[
            (0, 2, 0, 1),
            (2, 4, 0, 1),
            (0, 2, 1, 1),
            (2, 2, 1, 1),
            (4, 2, 1, 1),
            (0, 3, 2, 1),
            (3, 3, 2, 1),
        ],
    )
    figure_mechanistic_relate(plot.axes[2], plot.axes[3], plot.axes[4])
    figure_predict_effector(plot.axes[5:])
    plot.add_subplot_labels(ax_relative=True)
    plot.fig.tight_layout(pad=0, w_pad=0, h_pad=1)
    return plot.fig


def figure_mechanistic_relate(ax_0, ax_1, ax_2):
    zohar = Zohar()
    if UPDATE_CACHE["zohar"]:
        detection_signal = zohar.get_detection_signal()
        opts = assemble_options(detection_signal)
        x, ctx = optimize_loss(detection_signal, **opts, return_reshaped_params=True)
        Rtot = Rtot_to_df(x["Rtot"], detection_signal, rcps=list(opts["rcps"]))
        Rtot.to_csv(CACHE_DIR / "zohar_Rtot.csv")
    else:
        Rtot = pd.read_csv(CACHE_DIR / "zohar_Rtot.csv").set_index(
            ["Sample", "Antigen"], drop=True
        )
    Rtot = Rtot.xs("S", level="Antigen")

    fucose = compute_fucose_ratio(Rtot)
    df_meta = zohar.get_metadata()
    df_fucose_ratio = pd.merge(df_meta, fucose, on="Sample", how="inner")
    df_Rtot = pd.merge(Rtot, df_meta, on="Sample", how="inner")
    df_Rtot["total_afucosylated"] = df_Rtot["IgG1"] + df_Rtot["IgG3"]

    y = np.log10(df_fucose_ratio["FcR3A_S"] / df_fucose_ratio["FcR2A_S"] + 1)
    sns.scatterplot(
        y=y,
        x=df_fucose_ratio["fucose_inferred"],
        ax=ax_0,
        alpha=ALPHA,
    )
    ax_0.set_xlabel("Inferred IgG Fucosylation (%)")
    ax_0.set_ylabel(
        f"{LOG10_SYMBOL}"
        f"({DETECTION_DISPLAY_NAMES['FcR3A']} / {DETECTION_DISPLAY_NAMES['FcR2A']} + 1)"
    )
    r, p = pearsonr(df_fucose_ratio["fucose_inferred"], y)
    ax_0.text(
        0.05,
        0.05,
        r"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax_0.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
    )

    afucosylated_xlim = (-0.1e7, 1.4e7)
    afucosylated_xticks = np.linspace(0, 1.2e7, 4)
    afucosylated_xticklabels = [
        "0",
        r"4$\times 10^6$",
        r"8$\times 10^6$",
        r"12$\times 10^6$",
    ]

    sns.scatterplot(
        data=df_Rtot,
        y="ADNKA_CD107a_S",
        x="total_afucosylated",
        ax=ax_1,
        alpha=ALPHA,
    )
    ax_1.set_xlabel("Inferred afucosylated IgG")
    ax_1.set_ylabel(r"ADNKA (CD107a)")
    ax_1.set_xticks(afucosylated_xticks)
    ax_1.set_xticklabels(afucosylated_xticklabels)
    ax_1.set_xlim(afucosylated_xlim)
    r, p = pearsonr(df_Rtot["total_afucosylated"], df_Rtot["ADNKA_CD107a_S"])
    ax_1.text(
        0.05,
        0.86,
        r"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax_1.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
    )

    sns.scatterplot(
        data=df_Rtot,
        y="ADNKA_MIP1b_S",
        x="total_afucosylated",
        ax=ax_2,
        alpha=ALPHA,
    )
    ax_2.set_xlabel("Inferred afucosylated IgG")
    ax_2.set_ylabel(r"ADNKA (MIP1b)")
    ax_2.set_xticks(afucosylated_xticks)
    ax_2.set_xlim(afucosylated_xlim)
    ax_2.set_xticklabels(afucosylated_xticklabels)
    r, p = pearsonr(df_Rtot["total_afucosylated"], df_Rtot["ADNKA_MIP1b_S"])
    ax_2.text(
        0.05,
        0.86,
        r"r = " + str(round(r, 2)),
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax_2.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
    )


def figure_predict_effector(axes):
    zohar = Zohar()
    Rtot = (
        pd.read_csv(CACHE_DIR / "zohar_Rtot.csv")
        .set_index(["Sample", "Antigen"], drop=True)
        .xs("S", level="Antigen")
    )
    df_meta = zohar.get_metadata()
    detection_signal = data_to_df(zohar.get_detection_signal()).xs(
        "S", level="Antigen"
    )[["IgG1", "IgG3"]]
    effector_cols = ["ADCP_S", "ADNKA_CD107a_S", "ADNKA_MIP1b_S"]
    accuracy_predictions(
        axes[0],
        detection_signal,
        Rtot,
        {col: df_meta[col] for col in effector_cols},
        xticklabel_mappings={
            "ADCP_S": "ADCP",
            "ADNKA_CD107a_S": "ADNKA (CD107a)",
            "ADNKA_MIP1b_S": "ADNKA (MIP1b)",
        },
        k=8,
        title="Zohar et al.",
    )

    alter = Alter()
    effector_functions = alter.get_effector_functions()
    detection_signal = data_to_df(alter.get_detection_signal(select_ligs=False)).xs(
        "gp120.SF162", level="Antigen"
    )[["IgG1", "IgG2", "IgG3", "IgG4"]]
    # drop nans in effector_functions
    effector_cols = ["ADCC", "ADNP", "CD107a", "MIP1b", "IFNy"]
    Rtot = (
        pd.read_csv(ALTER_RTOT_CACHE_PATH)
        .set_index(["Sample", "Antigen"], drop=True)
        .xs("gp120.SF162", level="Antigen")
    )
    accuracy_predictions(
        axes[1],
        detection_signal,
        Rtot,
        {col: effector_functions[col] for col in effector_cols},
        xticklabel_mappings={
            "ADCC": "ADCC",
            "ADNP": "ADNP",
            "CD107a": "ADNKA (CD107a)",
            "MIP1b": "ADNKA (MIP1b)",
            "IFNy": "ADNKA (IFNy)",
        },
        k=8,
        title="Alter et al.",
    )


def accuracy_predictions(
    ax,
    detection_signal: pd.DataFrame,
    Rtot: pd.DataFrame,
    labelss: dict[str, pd.Series],
    xticklabel_mappings: dict[str, str] = None,
    k: int = 10,
    title=None,
):
    rows = []
    for label_col in labelss:
        detection_signal_labeled = detection_signal.copy()
        Rtot_labeled = Rtot.copy()
        labels = labelss[label_col].dropna()
        detection_signal_labeled = detection_signal_labeled.merge(
            labels, how="inner", on="Sample"
        )
        Rtot_labeled = Rtot_labeled.merge(labels, how="inner", on="Sample")

        X_detection = detection_signal_labeled.drop(columns=[label_col])
        y_detection = detection_signal_labeled[label_col]

        X_Rtot = Rtot_labeled.drop(columns=[label_col])
        y_Rtot = Rtot_labeled[label_col]

        # Initialize K-Fold cross-validation
        kf = RepeatedKFold(n_splits=k, n_repeats=10, random_state=42)

        # Cross-validation for detection_signal
        for train_index, test_index in kf.split(X_detection):
            X_train, X_test = (
                X_detection.iloc[train_index],
                X_detection.iloc[test_index],
            )
            y_train, y_test = (
                y_detection.iloc[train_index],
                y_detection.iloc[test_index],
            )

            # Standardize features using training data statistics
            X_train_scaled = (X_train - X_train.mean()) / X_train.std()
            X_train_scaled["intercept"] = 1
            betas = np.linalg.lstsq(X_train_scaled, y_train, rcond=None)[0]

            # Apply the same transformation to test data
            X_test_scaled = (X_test - X_train.mean()) / X_train.std()
            X_test_scaled["intercept"] = 1
            y_pred = X_test_scaled.dot(betas)

            r2 = r2_score(y_test, y_pred)

            rows.append(
                {
                    "label_col": label_col,
                    "r2": r2,
                    "Regressors": "Subclass detections",
                }
            )

        # Cross-validation for Rtot
        for train_index, test_index in kf.split(X_Rtot):
            X_train, X_test = X_Rtot.iloc[train_index], X_Rtot.iloc[test_index]
            y_train, y_test = y_Rtot.iloc[train_index], y_Rtot.iloc[test_index]

            # Standardize features using training data statistics
            X_train_scaled = (X_train - X_train.mean()) / X_train.std()
            X_train_scaled["intercept"] = 1
            betas = np.linalg.lstsq(X_train_scaled, y_train, rcond=None)[0]

            # Apply the same transformation to test data
            X_test_scaled = (X_test - X_train.mean()) / X_train.std()
            X_test_scaled["intercept"] = 1
            y_pred = X_test_scaled.dot(betas)

            r2 = r2_score(y_test, y_pred)
            rows.append(
                {
                    "label_col": label_col,
                    "r2": r2,
                    "Regressors": "Inferred Fc species",
                }
            )

    # Create DataFrame of accuracies
    accuracies = pd.DataFrame(rows)

    # Plot the accuracies
    sns.barplot(data=accuracies, x="label_col", y="r2", hue="Regressors", ax=ax)
    ax.set_ylabel("Regression cross-validation accuracy ($R^2$)")
    ax.set_xlabel("Effector function (response variable)")
    legend = ax.legend(loc="lower right", title="Regressors")
    legend.get_frame().set_alpha(0.85)
    if xticklabel_mappings is not None:
        ax.set_xticklabels(
            [xticklabel_mappings[label.get_text()] for label in ax.get_xticklabels()],
            rotation=25,
        )

    pairs = [
        ((label, "Subclass detections"), (label, "Inferred Fc species"))
        for label in labelss
    ]
    annotator = Annotator(
        ax,
        pairs,
        data=accuracies,
        x="label_col",
        y="r2",
        hue="Regressors",
    )
    annotate_mann_whitney(annotator)
    ax.set_title(title)
