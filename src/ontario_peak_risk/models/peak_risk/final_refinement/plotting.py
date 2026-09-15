"""Plots for XGBoost Peak-Risk final threshold refinement."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def precision_recall_f1_by_fold(
    fold_sweep: pd.DataFrame,
    selected_threshold: float,
):
    figures = []

    for fold, group in fold_sweep.groupby("fold", observed=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        group = group.sort_values("threshold")

        ax.plot(
            group["threshold"],
            group["precision"],
            label="Precision",
        )
        ax.plot(
            group["threshold"],
            group["recall"],
            label="Recall",
        )
        ax.plot(
            group["threshold"],
            group["f1"],
            label="F1",
        )
        ax.axvline(
            selected_threshold,
            linestyle="--",
            label=f"Selected = {selected_threshold:.2f}",
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title(f"XGBoost — Threshold Trade-off ({fold})")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        figures.append(fig)

    return figures


def false_negative_tradeoff(
    fold_summary: pd.DataFrame,
    selected_threshold: float,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        fold_summary["threshold"],
        fold_summary["max_missed_peak_rate_pct"],
        label="Worst-fold missed Peak rate",
    )
    ax.plot(
        fold_summary["threshold"],
        fold_summary["mean_false_alarm_rate_pct"],
        label="Mean false-alarm rate",
    )
    ax.axvline(
        selected_threshold,
        linestyle="--",
        label=f"Selected = {selected_threshold:.2f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Operational Trade-off — Missed Peaks vs False Alarms")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def stability_plot(
    fold_summary: pd.DataFrame,
    selected_threshold: float,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        fold_summary["threshold"],
        fold_summary["worst_fold_f1"],
        label="Worst-fold F1",
    )
    ax.plot(
        fold_summary["threshold"],
        fold_summary["mean_f1"],
        label="Mean F1",
    )
    ax.plot(
        fold_summary["threshold"],
        fold_summary["min_recall"],
        label="Minimum fold Recall",
    )
    ax.axvline(
        selected_threshold,
        linestyle="--",
        label=f"Selected = {selected_threshold:.2f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Threshold Stability Across Validation Folds")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def m9w_tradeoff_plot(
    stress: pd.DataFrame,
    selected_threshold: float,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        stress["threshold"],
        stress["precision"],
        label="Precision",
    )
    ax.plot(
        stress["threshold"],
        stress["recall"],
        label="Recall",
    )
    ax.plot(
        stress["threshold"],
        stress["f1"],
        label="F1",
    )
    ax.axvline(
        selected_threshold,
        linestyle="--",
        label=f"Selected = {selected_threshold:.2f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("M9W-2023 Stress Case — Threshold Trade-off")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def selected_metric_by_horizon(
    horizon_metrics: pd.DataFrame,
    metric: str,
):
    fig, ax = plt.subplots(figsize=(11, 5))

    for fold, group in horizon_metrics.groupby("fold", observed=True):
        group = group.sort_values("horizon")
        ax.plot(
            group["horizon"],
            group[metric],
            marker="o",
            markersize=3,
            label=fold,
        )

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(
        f"Selected Threshold — {metric.replace('_', ' ').title()} by Horizon"
    )
    ax.set_xticks(range(1, 25))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
