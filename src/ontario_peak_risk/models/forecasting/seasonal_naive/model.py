"""Daily Seasonal Naive baseline for 24-hour electricity-demand forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_daily_seasonal_naive_predictions(
    forecast_origins: pd.DataFrame,
    observed_demand: pd.DataFrame,
    horizons: int = 24,
    seasonal_period_hours: int = 24,
    target_prefix: str = "target_h",
    group_column: str = "fsa",
    origin_column: str = "forecast_origin",
    observation_time_column: str = "timestamp",
    demand_column: str = "total_consumption_kwh",
) -> pd.DataFrame:
    """
    Predict each future hour using demand from the same hour one day earlier.

    For target time `forecast_origin + h`, the prediction is the observed demand
    at `forecast_origin + h - 24 hours`.

    For h=1..24 this reference timestamp is never later than forecast_origin,
    so the baseline does not use future demand.
    """
    required_origins = {group_column, origin_column}
    missing_origins = sorted(required_origins.difference(forecast_origins.columns))
    if missing_origins:
        raise ValueError(f"Forecast origins are missing columns: {missing_origins}")

    required_observed = {
        group_column,
        observation_time_column,
        demand_column,
    }
    missing_observed = sorted(required_observed.difference(observed_demand.columns))
    if missing_observed:
        raise ValueError(f"Observed demand is missing columns: {missing_observed}")

    if horizons <= 0:
        raise ValueError("horizons must be greater than zero.")

    if seasonal_period_hours <= 0:
        raise ValueError("seasonal_period_hours must be greater than zero.")

    source = observed_demand[
        [group_column, observation_time_column, demand_column]
    ].copy()

    source[observation_time_column] = pd.to_datetime(
        source[observation_time_column],
        errors="coerce",
    )

    if source[observation_time_column].isna().any():
        raise ValueError("Observed demand contains invalid timestamps.")

    if source.duplicated([group_column, observation_time_column]).any():
        raise ValueError("Observed demand contains duplicated FSA + timestamp keys.")

    lookup = source.set_index(
        [group_column, observation_time_column]
    )[demand_column]

    predictions = forecast_origins[[group_column, origin_column]].copy()
    predictions[origin_column] = pd.to_datetime(
        predictions[origin_column],
        errors="coerce",
    )

    for horizon in range(1, horizons + 1):
        target_time = (
            predictions[origin_column]
            + pd.to_timedelta(horizon, unit="h")
        )
        reference_time = (
            target_time
            - pd.to_timedelta(seasonal_period_hours, unit="h")
        )

        keys = pd.MultiIndex.from_arrays(
            [
                predictions[group_column].to_numpy(),
                reference_time.to_numpy(),
            ],
            names=[group_column, observation_time_column],
        )

        predictions[f"{target_prefix}{horizon:02d}"] = (
            lookup.reindex(keys).to_numpy()
        )

    return predictions


def validate_daily_seasonal_naive_availability(
    predictions: pd.DataFrame,
    target_prefix: str = "target_h",
) -> pd.DataFrame:
    """Summarize whether all Seasonal Naive predictions are available."""
    columns = sorted(
        column
        for column in predictions.columns
        if column.startswith(target_prefix)
    )

    return pd.DataFrame(
        [
            {
                "target": column,
                "available_predictions": int(predictions[column].notna().sum()),
                "missing_predictions": int(predictions[column].isna().sum()),
                "coverage_pct": float(predictions[column].notna().mean() * 100),
                "status": (
                    "PASS"
                    if predictions[column].notna().all()
                    else "FAIL"
                ),
            }
            for column in columns
        ]
    )
