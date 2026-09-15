"""Classification diagnostics shared inside one Peak-Risk model branch."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    actual,
    predicted,
    probability,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    probability = np.asarray(probability, dtype=float)

    result = {
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
        "n_observations": int(len(actual)),
        "positive_rate_pct": float(actual.mean() * 100),
        "predicted_positive_rate_pct": float(predicted.mean() * 100),
        "brier_score": float(
            brier_score_loss(actual, probability)
        ),
    }

    if len(np.unique(actual)) == 2:
        result["pr_auc"] = float(
            average_precision_score(actual, probability)
        )
        result["roc_auc"] = float(
            roc_auc_score(actual, probability)
        )
    else:
        result["pr_auc"] = np.nan
        result["roc_auc"] = np.nan

    return result


def threshold_sweep(
    predictions: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Diagnostic threshold sweep only.

    The common model-comparison threshold remains 0.50.
    """
    if thresholds is None:
        thresholds = [
            round(value, 2)
            for value in np.arange(0.05, 0.96, 0.05)
        ]

    rows = []
    actual = predictions["actual"].astype(int).to_numpy()
    probability = predictions["probability"].astype(float).to_numpy()

    for threshold in thresholds:
        predicted = (probability >= threshold).astype(int)
        metrics = classification_metrics(
            actual,
            predicted,
            probability,
        )
        rows.append(
            {
                "threshold": float(threshold),
                **metrics,
            }
        )

    return pd.DataFrame(rows)
