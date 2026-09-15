"""Leakage-safe horizon-specific features for the Logistic Regression baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_logistic_horizon_features(
    origins: pd.DataFrame,
    observed_frame: pd.DataFrame,
    horizon: int,
    group_column: str = "fsa",
    origin_column: str = "forecast_origin",
    observation_time_column: str = "timestamp",
    demand_column: str = "total_consumption_kwh",
    season_column: str = "season",
) -> pd.DataFrame:
    """
    Build simple, interpretable features available at forecast origin.

    Features:
    - FSA;
    - target season;
    - target-hour/day/month cyclic encodings;
    - target weekend indicator;
    - same target hour's consumption 24 hours earlier;
    - same target hour's consumption 168 hours earlier.

    For horizons 1..24, the 24-hour reference timestamp is never later than
    the forecast origin, so it remains leakage-safe.
    """
    if not 1 <= horizon <= 24:
        raise ValueError("horizon must be between 1 and 24.")

    required = {
        group_column,
        observation_time_column,
        demand_column,
        season_column,
    }
    missing = sorted(required.difference(observed_frame.columns))
    if missing:
        raise ValueError(f"Observed frame is missing required columns: {missing}")

    source = observed_frame[
        [
            group_column,
            observation_time_column,
            demand_column,
            season_column,
        ]
    ].copy()

    source[observation_time_column] = pd.to_datetime(
        source[observation_time_column],
        errors="coerce",
    )

    if source.duplicated([group_column, observation_time_column]).any():
        raise ValueError("Observed frame contains duplicated FSA + timestamp keys.")

    demand_lookup = source.set_index(
        [group_column, observation_time_column]
    )[demand_column]

    season_lookup = source.set_index(
        [group_column, observation_time_column]
    )[season_column]

    result = origins[[group_column, origin_column]].copy()
    result[origin_column] = pd.to_datetime(
        result[origin_column],
        errors="coerce",
    )

    target_timestamp = (
        result[origin_column]
        + pd.to_timedelta(horizon, unit="h")
    )

    result["target_timestamp"] = target_timestamp

    hour = target_timestamp.dt.hour
    weekday = target_timestamp.dt.weekday
    month = target_timestamp.dt.month

    result["target_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["target_hour_cos"] = np.cos(2 * np.pi * hour / 24)

    result["target_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["target_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    result["target_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    result["target_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    result["target_is_weekend"] = (weekday >= 5).astype("int8")

    target_keys = pd.MultiIndex.from_arrays(
        [
            result[group_column].to_numpy(),
            target_timestamp.to_numpy(),
        ],
        names=[group_column, observation_time_column],
    )

    result["target_season"] = season_lookup.reindex(target_keys).to_numpy()

    for lag_hours, column_name in [
        (24, "previous_day_same_hour_consumption"),
        (168, "previous_week_same_hour_consumption"),
    ]:
        reference_timestamp = (
            target_timestamp
            - pd.to_timedelta(lag_hours, unit="h")
        )

        keys = pd.MultiIndex.from_arrays(
            [
                result[group_column].to_numpy(),
                reference_timestamp.to_numpy(),
            ],
            names=[group_column, observation_time_column],
        )

        result[column_name] = demand_lookup.reindex(keys).to_numpy()

    return result


def logistic_feature_columns() -> tuple[list[str], list[str]]:
    """Return numeric and categorical predictor columns."""
    numeric = [
        "target_hour_sin",
        "target_hour_cos",
        "target_weekday_sin",
        "target_weekday_cos",
        "target_month_sin",
        "target_month_cos",
        "target_is_weekend",
        "previous_day_same_hour_consumption",
        "previous_week_same_hour_consumption",
    ]

    categorical = [
        "fsa",
        "target_season",
    ]

    return numeric, categorical
