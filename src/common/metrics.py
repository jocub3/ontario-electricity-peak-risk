"""Forecast accuracy metrics, shared across the team so results are comparable.

Every function takes ``actual``/``prediction`` arrays (or a long-format forecast dataframe with
those columns) and drops any pair where ``actual`` is NaN first, since the last horizons of the
last fold have no future data to score against (see `feature_engineering.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _drop_missing_actual(actual: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(actual)
    return actual[mask], prediction[mask]


def mae(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual, prediction = _drop_missing_actual(np.asarray(actual), np.asarray(prediction))
    return float(np.mean(np.abs(actual - prediction)))


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual, prediction = _drop_missing_actual(np.asarray(actual), np.asarray(prediction))
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))


def wape(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Weighted absolute percentage error: total absolute error over total actual consumption."""
    actual, prediction = _drop_missing_actual(np.asarray(actual), np.asarray(prediction))
    return float(np.sum(np.abs(actual - prediction)) / np.sum(np.abs(actual)))


def mape(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Mean absolute percentage error. Consumption never sits near zero in this dataset, but rows
    with exactly zero actual are still dropped to avoid a division by zero."""
    actual, prediction = _drop_missing_actual(np.asarray(actual), np.asarray(prediction))
    nonzero = actual != 0
    actual, prediction = actual[nonzero], prediction[nonzero]
    return float(np.mean(np.abs((actual - prediction) / actual)) * 100)


def summary_metrics(forecast: pd.DataFrame) -> dict[str, float]:
    """MAE, WAPE, RMSE, and MAPE over an entire long-format forecast dataframe."""
    actual, prediction = forecast["actual"].to_numpy(), forecast["prediction"].to_numpy()
    return {
        "mae": mae(actual, prediction),
        "wape": wape(actual, prediction),
        "rmse": rmse(actual, prediction),
        "mape": mape(actual, prediction),
    }


def metrics_by_group(forecast: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """MAE, WAPE, RMSE, and MAPE grouped by an arbitrary column (e.g. `horizon`, `fsa`)."""
    rows = []
    for group_value, group in forecast.groupby(group_column):
        rows.append({group_column: group_value, **summary_metrics(group)})
    return pd.DataFrame(rows).sort_values(group_column).reset_index(drop=True)


def metrics_by_horizon(forecast: pd.DataFrame) -> pd.DataFrame:
    """MAE, WAPE, RMSE, and MAPE per forecast horizon (h+1 .. h+24)."""
    return metrics_by_group(forecast, "horizon")


def classification_summary_metrics(forecast: pd.DataFrame) -> dict[str, float]:
    """PR-AUC, ROC-AUC, Brier score, precision, recall, and F1 over a peak-risk forecast
    dataframe (columns ``actual_peak``, ``peak_probability``, ``predicted_peak``)."""
    actual = forecast["actual_peak"].to_numpy()
    probability = forecast["peak_probability"].to_numpy()
    predicted = forecast["predicted_peak"].to_numpy()
    return {
        "pr_auc": average_precision_score(actual, probability),
        "roc_auc": roc_auc_score(actual, probability),
        "brier": brier_score_loss(actual, probability),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
    }


def classification_metrics_by_group(forecast: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Classification metrics grouped by an arbitrary column (e.g. ``fold``, ``fsa``)."""
    rows = []
    for group_value, group in forecast.groupby(group_column):
        rows.append({group_column: group_value, **classification_summary_metrics(group)})
    return pd.DataFrame(rows).sort_values(group_column).reset_index(drop=True)


def top_k_capture_rate(forecast: pd.DataFrame, k_fraction: float = 0.025) -> float:
    """Share of the true peaks caught within the top ``k_fraction`` highest-probability rows.

    Defaults to 2.5%, matching the peak definition, so a perfect model scores 1.0.
    """
    n_top = max(1, int(len(forecast) * k_fraction))
    top = forecast.nlargest(n_top, "peak_probability")
    total_actual_peaks = forecast["actual_peak"].sum()
    return top["actual_peak"].sum() / total_actual_peaks


def calibration_table(forecast: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs. actual peak rate, in ``n_bins`` equal-width bins.

    A well-calibrated model has ``actual_rate`` close to ``mean_predicted`` in every bin.
    """
    binned = forecast[["peak_probability", "actual_peak"]].copy()
    binned["bin"] = pd.cut(binned["peak_probability"], bins=n_bins, include_lowest=True)
    return (
        binned.groupby("bin", observed=True)
        .agg(
            mean_predicted=("peak_probability", "mean"),
            actual_rate=("actual_peak", "mean"),
            n=("actual_peak", "size"),
        )
        .reset_index()
    )
