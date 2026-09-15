"""Common metric definitions for Forecasting and Peak-Risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def forecasting_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate common regression metrics."""
    actual = _as_float_array(y_true)
    predicted = _as_float_array(y_pred)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if actual.size == 0:
        raise ValueError("No valid observations are available for forecasting metrics.")

    errors = predicted - actual

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = actual != 0
    mape = (
        np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
        if nonzero.any()
        else np.nan
    )

    denominator = np.abs(actual).sum()
    wape = (
        np.abs(actual - predicted).sum() / denominator * 100
        if denominator != 0
        else np.nan
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "wape": float(wape),
        "bias": float(errors.mean()),
        "n_observations": int(actual.size),
    }


def peak_risk_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Calculate common binary-classification metrics."""
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)

    result = {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "n_observations": int(len(actual)),
        "positive_rate_pct": float(actual.mean() * 100),
    }

    if y_score is not None and len(np.unique(actual)) == 2:
        score = _as_float_array(y_score)
        result["pr_auc"] = float(average_precision_score(actual, score))
        result["roc_auc"] = float(roc_auc_score(actual, score))
    else:
        result["pr_auc"] = np.nan
        result["roc_auc"] = np.nan

    return result


def metric_dictionary() -> pd.DataFrame:
    """Return the official project metric definitions."""
    return pd.DataFrame(
        [
            ["forecasting", "MAE", "primary", "lower_is_better", "Mean absolute forecast error in kWh."],
            ["forecasting", "RMSE", "primary", "lower_is_better", "Root mean squared forecast error in kWh."],
            ["forecasting", "MAPE", "primary", "lower_is_better", "Mean absolute percentage error for non-zero demand."],
            ["forecasting", "WAPE", "secondary", "lower_is_better", "Absolute error relative to total observed demand."],
            ["forecasting", "Bias", "secondary", "closer_to_zero", "Mean signed forecast error."],
            ["peak_risk", "Precision", "primary", "higher_is_better", "Share of predicted Peaks that are true Peaks."],
            ["peak_risk", "Recall", "primary", "higher_is_better", "Share of true Peaks correctly detected."],
            ["peak_risk", "F1", "primary", "higher_is_better", "Harmonic mean of Precision and Recall."],
            ["peak_risk", "PR-AUC", "primary", "higher_is_better", "Precision-Recall area; emphasized for class imbalance."],
            ["peak_risk", "ROC-AUC", "secondary", "higher_is_better", "Area under the ROC curve."],
            ["peak_risk", "Balanced Accuracy", "secondary", "higher_is_better", "Average recall across both classes."],
        ],
        columns=["task", "metric", "role", "direction", "description"],
    )
