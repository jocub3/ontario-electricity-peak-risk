"""Comparison plots. Figures are saved and also returned for notebook display."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, output_dir: Path | None, filename: str):
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / filename,
            dpi=160,
            bbox_inches="tight",
        )


def forecasting_global_metrics_plot(
    table: pd.DataFrame,
    output_dir: Path | None = None,
):
    metrics = ["mae", "rmse"]
    available = [m for m in metrics if m in table.columns]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(table))
    width = 0.8 / len(available)

    for idx, metric in enumerate(available):
        ax.bar(
            x + idx * width,
            table[metric],
            width,
            label=metric.upper(),
        )

    ax.set_xticks(
        x + width * (len(available) - 1) / 2,
        table["model"],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Error (kWh)")
    ax.set_title("Forecasting — Global MAE and RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir, "forecasting_global_mae_rmse.png")
    return fig


def forecasting_percentage_metrics_plot(
    table: pd.DataFrame,
    output_dir: Path | None = None,
):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(table))
    width = 0.36

    ax.bar(x - width / 2, table["mape"], width, label="MAPE")
    if "wape" in table.columns:
        ax.bar(x + width / 2, table["wape"], width, label="WAPE")

    ax.set_xticks(x, table["model"], rotation=30, ha="right")
    ax.set_ylabel("Percentage Error")
    ax.set_title("Forecasting — Global Percentage Error")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir, "forecasting_global_percentage_errors.png")
    return fig


def metric_by_fold_plot(
    fold_table: pd.DataFrame,
    metric: str,
    title: str,
    output_dir: Path | None = None,
    filename: str = "metric_by_fold.png",
):
    pivot = fold_table.pivot(
        index="model",
        columns="fold",
        values=metric,
    )

    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig = ax.get_figure()
    _save(fig, output_dir, filename)
    return fig


def metric_by_horizon_plot(
    horizon_table: pd.DataFrame,
    metric: str,
    title: str,
    output_dir: Path | None = None,
    filename: str = "metric_by_horizon.png",
):
    fig, ax = plt.subplots(figsize=(12, 6))

    grouped = (
        horizon_table
        .groupby(["model", "horizon"], observed=True)[metric]
        .mean()
        .reset_index()
    )

    for model, group in grouped.groupby("model", observed=True):
        group = group.sort_values("horizon")
        ax.plot(
            group["horizon"],
            group[metric],
            marker="o",
            markersize=3,
            label=model,
        )

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticks(range(1, 25))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir, filename)
    return fig


def fsa_heatmap(
    fsa_table: pd.DataFrame,
    metric: str,
    fold: str,
    title: str,
    output_dir: Path | None = None,
    filename: str = "fsa_heatmap.png",
):
    subset = fsa_table[
        fsa_table["fold"].eq(fold)
    ].copy()

    # Some model reports may contain more than one row
    # for the same Model × Fold × FSA combination.
    # Aggregate them before reshaping.
    aggregated = (
        subset
        .groupby(
            ["model", "fsa"],
            observed=True,
            as_index=False,
        )[metric]
        .mean()
    )

    pivot = aggregated.pivot(
        index="model",
        columns="fsa",
        values=metric,
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    image = ax.imshow(
        pivot.values,
        aspect="auto",
    )

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)

    ax.set_xlabel("FSA")
    ax.set_ylabel("Model")
    ax.set_title(title)

    fig.colorbar(
        image,
        ax=ax,
        label=metric.upper(),
    )

    fig.tight_layout()

    _save(
        fig,
        output_dir,
        filename,
    )

    return fig

def m9w_bar_plot(
    stress_table: pd.DataFrame,
    metric: str,
    title: str,
    output_dir: Path | None = None,
    filename: str = "m9w_stress.png",
):
    data = stress_table.sort_values(metric)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data["model"], data[metric])
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticklabels(
        data["model"],
        rotation=30,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir, filename)
    return fig


def peak_global_plot(
    table: pd.DataFrame,
    output_dir: Path | None = None,
):
    metrics = ["pr_auc", "f1", "recall", "precision"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(table))
    width = 0.8 / len(metrics)

    for idx, metric in enumerate(metrics):
        ax.bar(
            x + idx * width,
            table[metric],
            width,
            label=metric.upper(),
        )

    ax.set_xticks(
        x + width * (len(metrics) - 1) / 2,
        table["model"],
        rotation=30,
        ha="right",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Peak-Risk — Global Model Performance")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir, "peak_risk_global_metrics.png")
    return fig


def calibration_comparison_plot(
    table: pd.DataFrame,
    output_dir: Path | None = None,
):
    available = table.dropna(subset=["brier_score"]).copy()
    if available.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(available["model"], available["brier_score"])
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title("Peak-Risk — Probability Calibration")
    ax.set_xticklabels(
        available["model"],
        rotation=30,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir, "peak_risk_brier_score.png")
    return fig


def runtime_plot(
    table: pd.DataFrame,
    title: str,
    output_dir: Path | None = None,
    filename: str = "runtime.png",
):
    data = table.dropna(
        subset=["runtime_minutes_evidence"]
    ).copy()

    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        data["model"],
        data["runtime_minutes_evidence"],
    )
    ax.set_ylabel("Runtime Evidence (minutes)")
    ax.set_title(title)
    ax.set_xticklabels(
        data["model"],
        rotation=30,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, output_dir, filename)
    return fig
