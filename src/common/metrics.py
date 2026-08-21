"""Evaluation metrics for the forecasting regressor and the peak-risk classifier."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --- Forecasting ---


def mae(actual, prediction) -> float:
    """Mean Absolute Error, in the same units as actual (kWh)."""
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    return float(np.mean(np.abs(actual - prediction)))


def wape(actual, prediction) -> float:
    """Weighted Absolute Percentage Error: total absolute error over total actual volume.

    Unlike a plain average of per-row percentage errors, this is a single ratio of sums, so
    it isn't distorted by rows where actual is near zero - the team's primary forecast metric.
    """
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    return float(np.sum(np.abs(actual - prediction)) / np.sum(np.abs(actual)))


def rmse(actual, prediction) -> float:
    """Root Mean Squared Error - like MAE but squares errors before averaging, so it penalizes
    large misses more than MAE does."""
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))


def mape(actual, prediction, epsilon: float = 1e-6) -> float:
    """Mean absolute percentage error, in %. epsilon guards against dividing by near-zero actuals."""
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    return float(np.mean(np.abs((actual - prediction) / np.maximum(np.abs(actual), epsilon))) * 100)


def forecast_metrics(actual, prediction) -> dict:
    """MAE, WAPE, RMSE, and MAPE together, for a quick per-fold summary."""
    return {
        "mae": mae(actual, prediction),
        "wape": wape(actual, prediction),
        "rmse": rmse(actual, prediction),
        "mape": mape(actual, prediction),
    }


def forecast_metrics_by_horizon(
    df: pd.DataFrame,
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    horizon_col: str = "horizon",
) -> pd.DataFrame:
    """forecast_metrics(), computed separately for each horizon."""
    rows = []
    for horizon, group in df.groupby(horizon_col):
        row = forecast_metrics(group[actual_col], group[prediction_col])
        row[horizon_col] = horizon
        rows.append(row)
    return pd.DataFrame(rows).set_index(horizon_col).sort_index()


# --- Peak-risk ---


def peak_risk_metrics(actual_peak, peak_probability, predicted_peak) -> dict:
    """PR-AUC (the team's primary metric), precision/recall/F1 at the chosen threshold,
    ROC-AUC, and Brier score, together for a quick per-fold summary."""
    return {
        "pr_auc": average_precision_score(actual_peak, peak_probability),
        "precision": precision_score(actual_peak, predicted_peak, zero_division=0),
        "recall": recall_score(actual_peak, predicted_peak, zero_division=0),
        "f1": f1_score(actual_peak, predicted_peak, zero_division=0),
        "roc_auc": roc_auc_score(actual_peak, peak_probability),
        "brier": brier_score_loss(actual_peak, peak_probability),
    }


def top_k_capture_rate(actual_peak, peak_probability, k: int) -> float:
    """Fraction of the true peaks that fall within the k highest predicted probabilities."""
    actual_peak = np.asarray(actual_peak)
    peak_probability = np.asarray(peak_probability)
    total_peaks = actual_peak.sum()
    if total_peaks == 0:
        return 0.0
    top_k_idx = np.argsort(-peak_probability)[:k]
    return float(actual_peak[top_k_idx].sum() / total_peaks)


def calibration_data(actual_peak, peak_probability, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs. observed peak frequency per bin, for a calibration plot."""
    observed, predicted = calibration_curve(
        actual_peak, peak_probability, n_bins=n_bins, strategy="quantile"
    )
    return pd.DataFrame({"predicted_probability": predicted, "observed_frequency": observed})


def pr_curve_data(actual_peak, peak_probability) -> pd.DataFrame:
    """Precision/recall at every distinct probability threshold, for a PR curve plot.

    sklearn's precision_recall_curve returns one more precision/recall pair than thresholds
    (the last pair, precision=1/recall=0, has no threshold that produces it) - padded with
    NaN so all 3 columns stay the same length.
    """
    precision, recall, thresholds = precision_recall_curve(actual_peak, peak_probability)
    thresholds = np.append(thresholds, np.nan)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": thresholds})
