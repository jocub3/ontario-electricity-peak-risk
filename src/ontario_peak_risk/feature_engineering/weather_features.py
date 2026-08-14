"""Weather-derived features justified by EDA findings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import make_feature_log


def _add_difference(
    frame: pd.DataFrame,
    column: str,
    lag: int,
) -> pd.Series:
    """Current weather minus weather observed at t-lag within FSA."""
    return (
        frame[column]
        - frame.groupby("fsa", observed=True)[column].shift(lag)
    )


def add_weather_features(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create weather change, rolling, and degree-based features.

    Current weather is intentionally retained because the operational project
    will use weather forecasts as future exogenous inputs.
    """
    output = frame.copy()
    weather = config["feature_engineering"]["weather"]
    rows = []

    temperature = weather["temperature_column"]
    humidity = weather["humidity_column"]
    pressure = weather["pressure_column"]

    difference_specs = [
        (temperature, weather["temperature_difference_lags"], "temperature"),
        (humidity, weather["humidity_difference_lags"], "humidity"),
        (pressure, weather["pressure_difference_lags"], "pressure"),
    ]

    for column, lags, prefix in difference_specs:
        if column not in output.columns:
            continue

        for lag in lags:
            name = f"{prefix}_change_{lag}h"
            output[name] = _add_difference(output, column, lag)

            rows.append(
                make_feature_log(
                    name,
                    "weather_change",
                    [column],
                    f"Change in {column} relative to {lag} hour(s) earlier.",
                    True,
                    True,
                    "historical_plus_current_exogenous",
                )
            )

    for column, prefix in [(temperature, "temperature"), (humidity, "humidity")]:
        if column not in output.columns:
            continue

        shifted = output.groupby("fsa", observed=True)[column].shift(1)
        grouped = shifted.groupby(output["fsa"], observed=True)

        for window in weather["weather_rolling_windows"]:
            name = f"{prefix}_rolling_mean_{window}h"
            output[name] = grouped.transform(
                lambda series: series.rolling(window, min_periods=window).mean()
            )

            rows.append(
                make_feature_log(
                    name,
                    "weather_rolling",
                    [column],
                    f"Mean {column} over the previous {window} hours, excluding the current hour.",
                    True,
                    True,
                    "historical_only",
                )
            )

    if temperature in output.columns:
        heating_base = float(weather["heating_base_c"])
        cooling_base = float(weather["cooling_base_c"])

        output["heating_degree_c"] = np.maximum(
            heating_base - output[temperature],
            0,
        )
        output["cooling_degree_c"] = np.maximum(
            output[temperature] - cooling_base,
            0,
        )

        rows.extend(
            [
                make_feature_log(
                    "heating_degree_c",
                    "weather_nonlinear",
                    [temperature],
                    f"Positive temperature deficit below {heating_base:g} °C.",
                    True,
                    True,
                    "current_exogenous",
                ),
                make_feature_log(
                    "cooling_degree_c",
                    "weather_nonlinear",
                    [temperature],
                    f"Positive temperature excess above {cooling_base:g} °C.",
                    True,
                    True,
                    "current_exogenous",
                ),
            ]
        )

    return output, pd.DataFrame(rows)
