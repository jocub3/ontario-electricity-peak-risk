"""Common model-evaluation helpers."""

from __future__ import annotations

import pandas as pd

from .metrics import forecasting_metrics, peak_risk_metrics


def evaluate_forecasting_by_horizon(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    target_prefix: str = "target_h",
) -> pd.DataFrame:
    """Evaluate a multi-horizon Forecasting output horizon by horizon."""
    target_columns = sorted(
        column for column in actual.columns if column.startswith(target_prefix)
    )

    rows = []

    for column in target_columns:
        if column not in predicted.columns:
            raise ValueError(f"Missing prediction column: {column}")

        metrics = forecasting_metrics(actual[column], predicted[column])
        horizon = int(column.replace(target_prefix, ""))

        rows.append(
            {
                "horizon": horizon,
                "target": column,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def evaluate_forecasting_global(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    target_prefix: str = "target_h",
) -> pd.DataFrame:
    """Evaluate all Forecasting horizons as one pooled set."""
    target_columns = sorted(
        column for column in actual.columns if column.startswith(target_prefix)
    )

    y_true = actual[target_columns].to_numpy().reshape(-1)
    y_pred = predicted[target_columns].to_numpy().reshape(-1)

    return pd.DataFrame([forecasting_metrics(y_true, y_pred)])


def evaluate_peak_risk(
    frame: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
    score_column: str | None = None,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate Peak-Risk globally or by specified groups."""
    groups = group_columns or []

    if not groups:
        groups_iter = [((), frame)]
    else:
        groups_iter = frame.groupby(groups, observed=True, sort=True)

    rows = []

    for key, group in groups_iter:
        score = group[score_column] if score_column is not None else None

        result = peak_risk_metrics(
            group[actual_column],
            group[predicted_column],
            score,
        )

        if groups:
            if not isinstance(key, tuple):
                key = (key,)
            prefix = dict(zip(groups, key))
        else:
            prefix = {}

        rows.append({**prefix, **result})

    return pd.DataFrame(rows)

def evaluate_forecasting_by_group(
    frame: pd.DataFrame,
    predicted: pd.DataFrame,
    group_columns: list[str],
    target_prefix: str = "target_h",
) -> pd.DataFrame:
    """
    Evaluate Forecasting performance by one or more grouping variables.

    Typical uses:
    - by FSA;
    - by validation fold;
    - by FSA and fold.

    The input frame must contain both grouping columns and actual
    target_hXX columns. The predicted DataFrame must have the same index
    and matching target_hXX prediction columns.
    """
    target_columns = sorted(
        column
        for column in frame.columns
        if column.startswith(target_prefix)
    )

    if not target_columns:
        raise ValueError(
            "No Forecasting target columns were found."
        )

    missing_predictions = [
        column
        for column in target_columns
        if column not in predicted.columns
    ]

    if missing_predictions:
        raise ValueError(
            "Missing prediction columns: "
            f"{missing_predictions}"
        )

    rows = []

    grouped = frame.groupby(
        group_columns,
        observed=True,
        sort=True,
    )

    for key, group in grouped:

        if not isinstance(key, tuple):
            key = (key,)

        group_info = dict(
            zip(
                group_columns,
                key,
            )
        )

        group_predictions = predicted.loc[
            group.index,
            target_columns,
        ]

        y_true = (
            group[target_columns]
            .to_numpy()
            .reshape(-1)
        )

        y_pred = (
            group_predictions[target_columns]
            .to_numpy()
            .reshape(-1)
        )

        metrics = forecasting_metrics(
            y_true,
            y_pred,
        )

        rows.append(
            {
                **group_info,
                **metrics,
            }
        )

    return pd.DataFrame(rows)

def probabilities_to_binary_predictions(
    probabilities,
    threshold: float = 0.50,
):
    """
    Convert Peak-Risk probabilities to binary predictions.

    The same threshold must be used for the initial comparison of all
    classifiers. Alternative operational thresholds may be optimized
    later using validation data only.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "Classification threshold must be between 0 and 1."
        )

    probabilities = pd.Series(
        probabilities,
        dtype="float64",
    )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    return (
        probabilities >= threshold
    ).astype("int8").to_numpy()

