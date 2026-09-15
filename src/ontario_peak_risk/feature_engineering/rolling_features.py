"""Leakage-safe historical rolling demand features."""

from __future__ import annotations

import pandas as pd

from .common import make_feature_log


def add_target_rolling_features(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add rolling target summaries based strictly on past observations.

    The target is shifted by one hour BEFORE each rolling calculation,
    ensuring that the current target value is never used.
    """
    output = frame.copy()
    target = config["feature_engineering"]["target_column"]
    rolling = config["feature_engineering"]["rolling"]
    windows = rolling["target_windows_hours"]
    statistics = rolling["statistics"]

    rows = []
    shifted = output.groupby("fsa", observed=True)[target].shift(1)

    for window in windows:
        grouped_shifted = shifted.groupby(output["fsa"], observed=True)

        for statistic in statistics:
            name = f"{target}_rolling_{statistic}_{window}h"

            if statistic == "mean":
                values = grouped_shifted.transform(
                    lambda series: series.rolling(window, min_periods=window).mean()
                )
            elif statistic == "std":
                values = grouped_shifted.transform(
                    lambda series: series.rolling(window, min_periods=window).std()
                )
            elif statistic == "min":
                values = grouped_shifted.transform(
                    lambda series: series.rolling(window, min_periods=window).min()
                )
            elif statistic == "max":
                values = grouped_shifted.transform(
                    lambda series: series.rolling(window, min_periods=window).max()
                )
            else:
                raise ValueError(f"Unsupported rolling statistic: {statistic}")

            output[name] = values

            rows.append(
                make_feature_log(
                    name,
                    "target_rolling",
                    [target],
                    f"{statistic.title()} of the previous {window} hourly consumption observations, excluding the current hour.",
                    True,
                    True,
                    "historical_only",
                    "low_if_shift_is_preserved",
                )
            )

    return output, pd.DataFrame(rows)
