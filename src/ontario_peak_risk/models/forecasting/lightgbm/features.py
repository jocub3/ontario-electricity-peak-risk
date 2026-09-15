"""Leakage-safe horizon-specific predictors for tree-based forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd


WEATHER_CANDIDATES = [
    "Temp (°C)",
    "Rel Hum (%)",
    "Stn Press (kPa)",
]

OPTIONAL_ORIGIN_CANDIDATES = [
    "reported_premise_count",
]


def _season_from_month(month: pd.Series) -> pd.Series:
    result = pd.Series(index=month.index, dtype="object")
    result.loc[month.isin([12, 1, 2])] = "Winter"
    result.loc[month.isin([3, 4, 5])] = "Spring"
    result.loc[month.isin([6, 7, 8])] = "Summer"
    result.loc[month.isin([9, 10, 11])] = "Fall"
    return result


def _build_origin_rolling_lookup(
    source: pd.DataFrame,
    group_column: str,
    time_column: str,
    demand_column: str,
) -> pd.DataFrame:
    work = source[[group_column, time_column, demand_column]].copy()
    work = work.sort_values([group_column, time_column], kind="stable")

    grouped = work.groupby(group_column, observed=True)[demand_column]

    work["origin_rolling_mean_24h"] = grouped.transform(
        lambda s: s.rolling(24, min_periods=12).mean()
    )
    work["origin_rolling_std_24h"] = grouped.transform(
        lambda s: s.rolling(24, min_periods=12).std()
    )
    work["origin_rolling_mean_168h"] = grouped.transform(
        lambda s: s.rolling(168, min_periods=24).mean()
    )

    return work.set_index([group_column, time_column])


def build_horizon_features(
    origins: pd.DataFrame,
    observed_frame: pd.DataFrame,
    horizon: int,
    group_column: str = "fsa",
    origin_column: str = "forecast_origin",
    observation_time_column: str = "timestamp",
    demand_column: str = "total_consumption_kwh",
) -> pd.DataFrame:
    """
    Build predictors using only information available at forecast origin
    plus deterministic target-time calendar information.
    """
    if not 1 <= int(horizon) <= 24:
        raise ValueError("horizon must be between 1 and 24.")

    required = {
        group_column,
        observation_time_column,
        demand_column,
    }
    missing = sorted(required.difference(observed_frame.columns))
    if missing:
        raise ValueError(f"Observed frame is missing required columns: {missing}")

    source_columns = [
        group_column,
        observation_time_column,
        demand_column,
        *[
            column
            for column in WEATHER_CANDIDATES + OPTIONAL_ORIGIN_CANDIDATES
            if column in observed_frame.columns
        ],
    ]

    source = observed_frame[source_columns].copy()
    source[observation_time_column] = pd.to_datetime(
        source[observation_time_column],
        errors="coerce",
    )

    if source[observation_time_column].isna().any():
        raise ValueError("Observed frame contains invalid timestamps.")

    if source.duplicated([group_column, observation_time_column]).any():
        raise ValueError("Observed frame contains duplicated FSA + timestamp keys.")

    source = source.sort_values(
        [group_column, observation_time_column],
        kind="stable",
    ).reset_index(drop=True)

    demand_lookup = source.set_index(
        [group_column, observation_time_column]
    )[demand_column]

    origin_lookup = source.set_index(
        [group_column, observation_time_column]
    )

    rolling_lookup = _build_origin_rolling_lookup(
        source,
        group_column=group_column,
        time_column=observation_time_column,
        demand_column=demand_column,
    )

    result = origins[[group_column, origin_column]].copy()
    result[origin_column] = pd.to_datetime(
        result[origin_column],
        errors="coerce",
    )

    target_timestamp = (
        result[origin_column]
        + pd.to_timedelta(int(horizon), unit="h")
    )

    hour = target_timestamp.dt.hour
    weekday = target_timestamp.dt.weekday
    month = target_timestamp.dt.month

    result["target_hour"] = hour.astype("int16")
    result["target_weekday"] = weekday.astype("int16")
    result["target_month"] = month.astype("int16")
    result["target_is_weekend"] = (weekday >= 5).astype("int8")
    result["target_season"] = _season_from_month(month)

    result["target_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["target_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["target_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["target_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    result["target_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    result["target_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    for lag_hours, name in [
        (24, "target_lag_24h"),
        (48, "target_lag_48h"),
        (168, "target_lag_168h"),
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

        result[name] = demand_lookup.reindex(keys).to_numpy()

    origin_keys = pd.MultiIndex.from_arrays(
        [
            result[group_column].to_numpy(),
            result[origin_column].to_numpy(),
        ],
        names=[group_column, observation_time_column],
    )

    for column in WEATHER_CANDIDATES + OPTIONAL_ORIGIN_CANDIDATES:
        if column in source.columns:
            result[f"origin__{column}"] = (
                origin_lookup[column].reindex(origin_keys).to_numpy()
            )

    for column in [
        "origin_rolling_mean_24h",
        "origin_rolling_std_24h",
        "origin_rolling_mean_168h",
    ]:
        result[column] = rolling_lookup[column].reindex(origin_keys).to_numpy()

    return result


def model_feature_columns(
    observed_frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    numeric = [
        "target_hour",
        "target_weekday",
        "target_month",
        "target_is_weekend",
        "target_hour_sin",
        "target_hour_cos",
        "target_weekday_sin",
        "target_weekday_cos",
        "target_month_sin",
        "target_month_cos",
        "target_lag_24h",
        "target_lag_48h",
        "target_lag_168h",
        "origin_rolling_mean_24h",
        "origin_rolling_std_24h",
        "origin_rolling_mean_168h",
    ]

    for column in WEATHER_CANDIDATES + OPTIONAL_ORIGIN_CANDIDATES:
        if column in observed_frame.columns:
            numeric.append(f"origin__{column}")

    categorical = ["fsa", "target_season"]
    return numeric, categorical


def feature_dictionary(
    observed_frame: pd.DataFrame,
) -> pd.DataFrame:
    numeric, categorical = model_feature_columns(observed_frame)
    rows = [
        {
            "feature": column,
            "type": "numeric",
            "availability": "forecast_origin_or_historical",
        }
        for column in numeric
    ]
    rows += [
        {
            "feature": column,
            "type": "categorical",
            "availability": "known_at_prediction_time",
        }
        for column in categorical
    ]
    return pd.DataFrame(rows)
