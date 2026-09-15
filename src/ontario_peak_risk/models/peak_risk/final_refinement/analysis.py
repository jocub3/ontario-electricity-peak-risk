"""Threshold analysis utilities for final XGBoost Peak-Risk refinement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def threshold_grid(
    minimum: float,
    maximum: float,
    step: float,
) -> np.ndarray:
    count = int(round((maximum - minimum) / step)) + 1
    values = minimum + np.arange(count) * step
    return np.round(values, 6)


def metrics_at_threshold(
    frame: pd.DataFrame,
    threshold: float,
) -> dict[str, float]:
    actual = frame["actual"].astype(int).to_numpy()
    probability = frame["probability"].astype(float).to_numpy()
    predicted = (probability >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    actual_peaks = tp + fn
    actual_no_peaks = tn + fp

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "missed_peak_rate_pct": (
            float(fn / actual_peaks * 100)
            if actual_peaks > 0
            else np.nan
        ),
        "false_alarm_rate_pct": (
            float(fp / actual_no_peaks * 100)
            if actual_no_peaks > 0
            else np.nan
        ),
        "predicted_peak_rate_pct": float(predicted.mean() * 100),
        "actual_peak_rate_pct": float(actual.mean() * 100),
        "n_observations": int(len(actual)),
    }


def threshold_sweep(
    predictions: pd.DataFrame,
    thresholds: np.ndarray,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    group_columns = group_columns or []

    if not group_columns:
        return pd.DataFrame(
            [
                metrics_at_threshold(predictions, threshold)
                for threshold in thresholds
            ]
        )

    rows = []

    for group_key, group in predictions.groupby(
        group_columns,
        observed=True,
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_values = dict(zip(group_columns, group_key))

        for threshold in thresholds:
            rows.append(
                {
                    **group_values,
                    **metrics_at_threshold(group, threshold),
                }
            )

    return pd.DataFrame(rows)


def fold_threshold_summary(
    fold_sweep: pd.DataFrame,
) -> pd.DataFrame:
    return (
        fold_sweep
        .groupby("threshold", observed=True)
        .agg(
            min_precision=("precision", "min"),
            mean_precision=("precision", "mean"),
            max_precision=("precision", "max"),
            min_recall=("recall", "min"),
            mean_recall=("recall", "mean"),
            max_recall=("recall", "max"),
            worst_fold_f1=("f1", "min"),
            mean_f1=("f1", "mean"),
            f1_range=("f1", lambda s: s.max() - s.min()),
            worst_fold_balanced_accuracy=("balanced_accuracy", "min"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            max_missed_peak_rate_pct=("missed_peak_rate_pct", "max"),
            mean_missed_peak_rate_pct=("missed_peak_rate_pct", "mean"),
            mean_false_alarm_rate_pct=("false_alarm_rate_pct", "mean"),
        )
        .reset_index()
    )


def select_operational_threshold(
    summary: pd.DataFrame,
    *,
    minimum_recall_each_fold: float,
    minimum_precision_each_fold: float,
) -> tuple[pd.Series, pd.DataFrame]:
    feasible = summary.loc[
        (summary["min_recall"] >= minimum_recall_each_fold)
        & (summary["min_precision"] >= minimum_precision_each_fold)
    ].copy()

    if feasible.empty:
        best_recall_candidates = summary.loc[
            summary["min_recall"] >= minimum_recall_each_fold
        ].copy()

        if not best_recall_candidates.empty:
            best_precision = best_recall_candidates[
                "min_precision"
            ].max()
        else:
            best_precision = float("nan")

        raise ValueError(
            "No threshold satisfies both operational constraints in every fold. "
            f"Required minimum Recall = {minimum_recall_each_fold:.2f}; "
            f"required minimum Precision = {minimum_precision_each_fold:.2f}. "
            f"Among thresholds satisfying the Recall requirement, the best "
            f"worst-fold Precision observed was {best_precision:.3f}. "
            "Review the operational constraints before selecting a threshold."
        )

    ranked = feasible.sort_values(
        [
            "worst_fold_f1",
            "min_recall",
            "mean_f1",
            "mean_precision",
            "threshold",
        ],
        ascending=[False, False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    return ranked.iloc[0], ranked


def evaluate_selected_threshold(
    predictions: pd.DataFrame,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    global_table = pd.DataFrame(
        [metrics_at_threshold(predictions, threshold)]
    )

    fold = threshold_sweep(
        predictions,
        np.array([threshold]),
        ["fold"],
    )

    fsa = threshold_sweep(
        predictions,
        np.array([threshold]),
        ["fold", "fsa"],
    )

    horizon = threshold_sweep(
        predictions,
        np.array([threshold]),
        ["fold", "horizon"],
    )

    return {
        "global": global_table,
        "fold": fold,
        "fsa": fsa,
        "horizon": horizon,
    }


def local_threshold_sensitivity(
    fold_summary: pd.DataFrame,
    selected_threshold: float,
    window: float,
) -> pd.DataFrame:
    return fold_summary.loc[
        fold_summary["threshold"].between(
            max(0.0, selected_threshold - window),
            min(1.0, selected_threshold + window),
        )
    ].copy()


def m9w_stress_sweep(
    predictions: pd.DataFrame,
    thresholds: np.ndarray,
    *,
    fold: str,
    fsa: str,
) -> pd.DataFrame:
    stress = predictions.loc[
        predictions["fold"].eq(fold)
        & predictions["fsa"].eq(fsa)
    ].copy()

    if stress.empty:
        return pd.DataFrame()

    return threshold_sweep(
        stress,
        thresholds,
    )
