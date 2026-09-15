"""Direct multi-horizon SARIMAX dataset preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_sarimax_exog(
    origins: pd.DataFrame,
    observed_frame: pd.DataFrame,
    horizon: int,
    origin_column: str = "forecast_origin",
    time_column: str = "timestamp",
    group_column: str = "fsa",
) -> pd.DataFrame:
    """
    Build compact exogenous variables for one direct forecast horizon.

    Only deterministic target-time calendar information and weather observed
    at the forecast origin are included. Future observed weather is excluded.
    """
    result = origins[[group_column, origin_column]].copy()
    result[origin_column] = pd.to_datetime(result[origin_column])

    target_timestamp = (
        result[origin_column]
        + pd.to_timedelta(horizon, unit="h")
    )

    hour = target_timestamp.dt.hour
    weekday = target_timestamp.dt.weekday
    month = target_timestamp.dt.month

    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    result["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    result["is_weekend"] = (weekday >= 5).astype("int8")

    source_columns = [
        group_column,
        time_column,
        *[
            column
            for column in [
                "Temp (°C)",
                "Rel Hum (%)",
                "Stn Press (kPa)",
                "reported_premise_count",
            ]
            if column in observed_frame.columns
        ],
    ]

    source = observed_frame[source_columns].copy()
    source[time_column] = pd.to_datetime(source[time_column])

    lookup = source.set_index([group_column, time_column])

    keys = pd.MultiIndex.from_arrays(
        [
            result[group_column].to_numpy(),
            result[origin_column].to_numpy(),
        ],
        names=[group_column, time_column],
    )

    rename_map = {
        "Temp (°C)": "origin_temperature",
        "Rel Hum (%)": "origin_humidity",
        "Stn Press (kPa)": "origin_pressure",
        "reported_premise_count": "origin_premise_count",
    }

    for source_column, output_column in rename_map.items():
        if source_column in source.columns:
            result[output_column] = (
                lookup[source_column].reindex(keys).to_numpy()
            )

    return result


def sarimax_exog_columns(frame: pd.DataFrame) -> list[str]:
    base = [
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
    ]
    optional = [
        "origin_temperature",
        "origin_humidity",
        "origin_pressure",
        "origin_premise_count",
    ]
    return base + [column for column in optional if column in frame.columns]
