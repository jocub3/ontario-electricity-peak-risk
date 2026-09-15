"""Training-window Peak-Risk target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _season_from_month(month: pd.Series) -> pd.Series:
    season = pd.Series(index=month.index, dtype="object")
    season.loc[month.isin([12, 1, 2])] = "Winter"
    season.loc[month.isin([3, 4, 5])] = "Spring"
    season.loc[month.isin([6, 7, 8])] = "Summer"
    season.loc[month.isin([9, 10, 11])] = "Fall"
    return season


def fit_peak_threshold_table(
    training_observations: pd.DataFrame,
    percentile: float = 0.975,
    *,
    target_column: str = "total_consumption_kwh",
) -> pd.DataFrame:
    """
    Fit Peak thresholds on training observations only.

    Threshold definition: quantile by FSA + season.
    """
    required = {"fsa", "timestamp", target_column}
    missing = sorted(required.difference(training_observations.columns))
    if missing:
        raise ValueError(f"Training observations missing: {missing}")

    frame = training_observations[
        ["fsa", "timestamp", target_column]
    ].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["season"] = _season_from_month(frame["timestamp"].dt.month)

    thresholds = (
        frame.groupby(["fsa", "season"], observed=True)[target_column]
        .quantile(float(percentile))
        .rename("peak_threshold")
        .reset_index()
    )

    if thresholds["peak_threshold"].isna().any():
        raise ValueError("Peak threshold table contains missing values.")

    return thresholds


def build_peak_labels(
    target_frame: pd.DataFrame,
    threshold_table: pd.DataFrame,
    horizon: int,
    *,
    forecast_target_prefix: str = "target_h",
) -> pd.DataFrame:
    """
    Label one direct horizon using a threshold table fitted on training only.

    `target_frame` is the common forecasting target matrix containing
    target_h01 ... target_h24.
    """
    horizon = int(horizon)
    target_column = f"{forecast_target_prefix}{horizon:02d}"

    if target_column not in target_frame.columns:
        raise ValueError(f"Missing target column: {target_column}")

    result = target_frame[
        ["fsa", "forecast_origin", target_column]
    ].copy()
    result["forecast_origin"] = pd.to_datetime(result["forecast_origin"])
    result["target_timestamp"] = (
        result["forecast_origin"]
        + pd.to_timedelta(horizon, unit="h")
    )
    result["season"] = _season_from_month(
        result["target_timestamp"].dt.month
    )

    result = result.merge(
        threshold_table,
        on=["fsa", "season"],
        how="left",
        validate="many_to_one",
    )

    if result["peak_threshold"].isna().any():
        raise ValueError(
            "Some horizon rows could not be matched to a training-derived "
            "FSA + season Peak threshold."
        )

    result["actual_peak"] = (
        result[target_column] > result["peak_threshold"]
    ).astype("int8")

    return result
