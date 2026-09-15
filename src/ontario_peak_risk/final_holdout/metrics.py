from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

def forecasting_metrics(actual, predicted) -> dict:
    actual = pd.Series(actual, dtype="float64")
    predicted = pd.Series(predicted, dtype="float64")
    residual = predicted - actual
    denom = actual.abs().replace(0, np.nan)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mape": float((residual.abs() / denom).mean() * 100),
        "wape": float(residual.abs().sum() / actual.abs().sum() * 100),
        "bias": float(residual.mean()),
        "n_observations": int(len(actual)),
    }

def classification_metrics(actual, predicted, probability) -> dict:
    actual = pd.Series(actual).astype(int)
    predicted = pd.Series(predicted).astype(int)
    probability = pd.Series(probability).astype(float)

    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()

    return {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "pr_auc": float(average_precision_score(actual, probability)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "brier_score": float(brier_score_loss(actual, probability)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "actual_peak_rate_pct": float(actual.mean() * 100),
        "predicted_peak_rate_pct": float(predicted.mean() * 100),
        "n_observations": int(len(actual)),
    }

def grouped_forecasting(predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(by, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(by, keys))
        row.update(forecasting_metrics(group["actual"], group["predicted"]))
        rows.append(row)
    return pd.DataFrame(rows)

def grouped_classification(predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(by, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(by, keys))
        row.update(
            classification_metrics(
                group["actual"],
                group["predicted"],
                group["probability"],
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)
