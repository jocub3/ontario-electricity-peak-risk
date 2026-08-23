"""Forecast accuracy metrics, shared across the team so results are comparable.

Every function takes ``actual``/``prediction`` arrays (or a long-format forecast dataframe with
those columns) and drops any pair where ``actual`` is NaN first, since the last horizons of the
last fold have no future data to score against (see `feature_engineering.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
