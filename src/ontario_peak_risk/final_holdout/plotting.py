from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Presentation palette
# ---------------------------------------------------------------------------

COLORS = {
    "actual": "#4C78A8",
    "predicted": "#F28E2B",
    "residual": "#4C78A8",
    "zero": "#555555",
    "reference": "#C44E52",
    "positive": "#59A14F",
    "negative": "#E15759",
    "neutral": "#B0B0B0",
    "curve": "#4C78A8",
    "baseline": "#888888",
}

def rf_actual_vs_predicted_daily(predictions: pd.DataFrame):
    data = predictions.copy()
    data["date"] = pd.to_datetime(data["target_timestamp"]).dt.floor("D")
    daily = data.groupby("date", observed=True)[["actual", "predicted"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["date"], daily["actual"], label="Actual")
    ax.plot(daily["date"], daily["predicted"], label="Predicted")
    ax.set_title("Random Forest — 2025 Daily Mean Actual vs Predicted")
    ax.set_ylabel("Consumption (kWh)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig

def rf_residual_distribution(predictions: pd.DataFrame):
    residual = predictions["predicted"] - predictions["actual"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residual.dropna(), bins=60)
    ax.axvline(0, linestyle="--")
    ax.set_title("Random Forest — 2025 Residual Distribution")
    ax.set_xlabel("Residual (Predicted - Actual)")
    fig.tight_layout()
    return fig

def rf_mae_by_horizon(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics["horizon"], metrics["mae"], marker="o")
    ax.set_xticks(range(1, 25))
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Random Forest — 2025 MAE by Horizon")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig

def xgb_pr_curve(predictions: pd.DataFrame):
    precision, recall, _ = precision_recall_curve(
        predictions["actual"], predictions["probability"]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("XGBoost — 2025 Precision-Recall Curve")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig

def xgb_roc_curve(predictions: pd.DataFrame):
    fpr, tpr, _ = roc_curve(predictions["actual"], predictions["probability"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("XGBoost — 2025 ROC Curve")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig

def xgb_confusion_matrix(predictions: pd.DataFrame):
    cm = confusion_matrix(predictions["actual"], predictions["predicted"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["No Peak", "Peak"])
    ax.set_yticks([0, 1], ["No Peak", "Peak"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("XGBoost — 2025 Confusion Matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig

def xgb_metric_by_horizon(metrics: pd.DataFrame, metric: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics["horizon"], metrics[metric], marker="o")
    ax.set_xticks(range(1, 25))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Horizon")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"XGBoost — 2025 {metric.replace('_', ' ').title()} by Horizon")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def rf_actual_vs_predicted_daily_enhanced(
    predictions: pd.DataFrame,
):
    """
    Presentation-ready daily Actual vs Predicted plot.
    """
    data = predictions.copy()

    data["date"] = (
        pd.to_datetime(data["target_timestamp"])
        .dt.floor("D")
    )

    daily = (
        data.groupby(
            "date",
            observed=True,
        )[["actual", "predicted"]]
        .mean()
        .reset_index()
    )

    daily["absolute_error"] = (
        daily["actual"] - daily["predicted"]
    ).abs()

    daily_mae = daily["absolute_error"].mean()

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    # Actual
    ax.plot(
        daily["date"],
        daily["actual"],
        color=COLORS["actual"],
        linewidth=1.7,
        label="Actual",
        zorder=3,
    )

    # Predicted
    ax.plot(
        daily["date"],
        daily["predicted"],
        color=COLORS["predicted"],
        linewidth=1.7,
        label="Predicted",
        zorder=3,
    )

    # Error area
    ax.fill_between(
        daily["date"],
        daily["actual"],
        daily["predicted"],
        color=COLORS["neutral"],
        alpha=0.12,
        label="Actual–Predicted gap",
        zorder=1,
    )

    ax.text(
        0.015,
        0.96,
        f"Daily-mean MAE: {daily_mae:,.0f} kWh",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.9,
        },
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Daily mean electricity consumption (kWh)"
    )

    ax.set_title(
        "Random Forest — 2025 Daily Mean Actual vs Predicted Demand"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.18,
    )

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper right",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig


def rf_residual_vs_predicted_enhanced(
    predictions: pd.DataFrame,
    sample_size: int = 50000,
):
    """
    Presentation-ready residual diagnostic plot.

    Residual = Actual - Predicted.
    Positive residual: model underprediction.
    Negative residual: model overprediction.
    """
    data = predictions[
        ["actual", "predicted"]
    ].dropna().copy()

    data["residual"] = (
        data["actual"]
        - data["predicted"]
    )

    median_residual = data["residual"].median()

    if len(data) > sample_size:
        sample = data.sample(
            sample_size,
            random_state=42,
        )
    else:
        sample = data

    positive = sample.loc[
        sample["residual"] >= 0
    ]

    negative = sample.loc[
        sample["residual"] < 0
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    # Underpredictions
    ax.scatter(
        positive["predicted"],
        positive["residual"],
        s=8,
        alpha=0.16,
        color=COLORS["positive"],
        label="Underprediction",
        rasterized=True,
    )

    # Overpredictions
    ax.scatter(
        negative["predicted"],
        negative["residual"],
        s=8,
        alpha=0.16,
        color=COLORS["negative"],
        label="Overprediction",
        rasterized=True,
    )

    # Perfect prediction reference
    ax.axhline(
        0,
        color=COLORS["zero"],
        linestyle="--",
        linewidth=1.8,
        label="Zero residual",
    )

    # Median residual
    ax.axhline(
        median_residual,
        color=COLORS["reference"],
        linestyle=":",
        linewidth=1.8,
        label=(
            f"Median residual = "
            f"{median_residual:,.0f} kWh"
        ),
    )

    ax.set_xlabel(
        "Predicted electricity demand (kWh)"
    )

    ax.set_ylabel(
        "Residual: Actual - Predicted (kWh)"
    )

    ax.set_title(
        "Random Forest — Residuals versus Predicted Demand on the 2025 Holdout"
    )

    ax.grid(
        alpha=0.15,
        linestyle="--",
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig


def xgb_confusion_matrix_enhanced(
    predictions: pd.DataFrame,
):
    """
    Presentation-ready confusion matrix with raw counts
    and row-normalized percentages.
    """
    cm = confusion_matrix(
        predictions["actual"],
        predictions["predicted"],
        labels=[0, 1],
    )

    row_totals = cm.sum(
        axis=1,
        keepdims=True,
    )

    cm_pct = (
        cm
        / row_totals
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 6)
    )

    image = ax.imshow(
        cm_pct,
        cmap="Blues",
        vmin=0,
        vmax=100,
    )

    labels = [
        ["True Negative", "False Positive"],
        ["False Negative", "True Positive"],
    ]

    for i in range(2):
        for j in range(2):

            percentage = cm_pct[i, j]
            count = cm[i, j]

            text_color = (
                "white"
                if percentage >= 55
                else "black"
            )

            ax.text(
                j,
                i,
                (
                    f"{labels[i][j]}\n"
                    f"{count:,}\n"
                    f"{percentage:.1f}%"
                ),
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
                fontweight="bold",
            )

    ax.set_xticks(
        [0, 1],
        ["No Peak", "Peak"],
    )

    ax.set_yticks(
        [0, 1],
        ["No Peak", "Peak"],
    )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    ax.set_title(
        "XGBoost — 2025 Peak-Risk Confusion Matrix"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Percentage within actual class (%)"
    )

    fig.tight_layout()

    return fig


def xgb_pr_curve_enhanced(
    predictions: pd.DataFrame,
    operational_threshold: float = 0.06,
):
    """
    Presentation-ready Precision-Recall curve.
    """
    y_true = predictions["actual"].to_numpy()
    probability = predictions[
        "probability"
    ].to_numpy()

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probability,
        )
    )

    average_precision = (
        average_precision_score(
            y_true,
            probability,
        )
    )

    prevalence = y_true.mean()

    # Find point nearest operational threshold.
    idx = np.argmin(
        np.abs(
            thresholds
            - operational_threshold
        )
    )

    operating_precision = precision[idx]
    operating_recall = recall[idx]

    fig, ax = plt.subplots(
        figsize=(7.5, 6.5)
    )

    ax.plot(
        recall,
        precision,
        color=COLORS["curve"],
        linewidth=2.2,
        label=(
            f"Precision-Recall curve "
            f"(AP = {average_precision:.3f})"
        ),
    )

    # Random / prevalence baseline
    ax.axhline(
        prevalence,
        color=COLORS["baseline"],
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Peak prevalence = "
            f"{prevalence:.1%}"
        ),
    )

    # Operational point
    ax.scatter(
        operating_recall,
        operating_precision,
        s=85,
        color=COLORS["reference"],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label=(
            f"Operational threshold = "
            f"{operational_threshold:.2f}"
        ),
    )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.set_title(
        "XGBoost — 2025 Precision-Recall Curve"
    )

    ax.grid(
        alpha=0.18,
        linestyle="--",
    )

    ax.legend(
        frameon=False,
        loc="lower left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig


def xgb_roc_curve_enhanced(
    predictions: pd.DataFrame,
    operational_threshold: float = 0.06,
):
    """
    Presentation-ready ROC curve.
    """
    y_true = predictions["actual"].to_numpy()

    probability = predictions[
        "probability"
    ].to_numpy()

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probability,
    )

    auc_value = roc_auc_score(
        y_true,
        probability,
    )

    # Closest point to operational threshold
    idx = np.argmin(
        np.abs(
            thresholds
            - operational_threshold
        )
    )

    operating_fpr = fpr[idx]
    operating_tpr = tpr[idx]

    fig, ax = plt.subplots(
        figsize=(7.5, 6.5)
    )

    ax.plot(
        fpr,
        tpr,
        color=COLORS["curve"],
        linewidth=2.2,
        label=f"ROC curve (AUC = {auc_value:.3f})",
    )

    # Random classifier
    ax.plot(
        [0, 1],
        [0, 1],
        color=COLORS["baseline"],
        linestyle="--",
        linewidth=1.5,
        label="Random classifier",
    )

    # Operational threshold
    ax.scatter(
        operating_fpr,
        operating_tpr,
        s=85,
        color=COLORS["reference"],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label=(
            f"Operational threshold = "
            f"{operational_threshold:.2f}"
        ),
    )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate (Recall)"
    )

    ax.set_title(
        "XGBoost — 2025 ROC Curve"
    )

    ax.grid(
        alpha=0.18,
        linestyle="--",
    )

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig



