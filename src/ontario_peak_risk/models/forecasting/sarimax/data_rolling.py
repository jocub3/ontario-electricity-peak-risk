"""Efficient rolling-origin exogenous preparation for SARIMAX.

This module intentionally does not replace the original ``data.py``.
It supports the revised SARIMAX implementation in ``run_model_rolling.py``.

The revised model uses only predictors that are deterministically known for
future timestamps. Observed future weather is excluded to prevent leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_calendar_exog(
    timestamps: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build deterministic calendar regressors for SARIMAX."""
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))

    hour = ts.hour
    weekday = ts.weekday
    month = ts.month

    return pd.DataFrame(
        {
            # Explicit intercept. The revised SARIMAX runner sets trend="n"
            # when the selected specification originally used trend="c".
            # Keeping the intercept in exog avoids statsmodels treating
            # single-row rolling updates as an accidental constant exog column.
            "intercept": 1.0,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
            "is_weekend": (weekday >= 5).astype("int8"),
        },
        index=ts,
    )


def prepare_observed_series(
    feature_dataset: pd.DataFrame,
    fsa: str,
    *,
    time_column: str = "timestamp",
    group_column: str = "fsa",
    demand_column: str = "total_consumption_kwh",
) -> pd.DataFrame:
    """Return one clean, ordered observed-demand series for an FSA."""
    required = {time_column, group_column, demand_column}
    missing = sorted(required.difference(feature_dataset.columns))
    if missing:
        raise ValueError(
            f"Feature dataset is missing required columns: {missing}"
        )

    frame = (
        feature_dataset.loc[
            feature_dataset[group_column].eq(fsa),
            [group_column, time_column, demand_column],
        ]
        .copy()
    )

    frame[time_column] = pd.to_datetime(
        frame[time_column],
        errors="coerce",
    )

    if frame[time_column].isna().any():
        raise ValueError(f"{fsa}: invalid timestamps were found.")

    if frame.duplicated([group_column, time_column]).any():
        duplicates = int(
            frame.duplicated([group_column, time_column]).sum()
        )
        raise ValueError(
            f"{fsa}: {duplicates} duplicated FSA + timestamp keys were found."
        )

    frame = (
        frame
        .sort_values(time_column, kind="stable")
        .reset_index(drop=True)
    )

    if frame[demand_column].isna().any():
        raise ValueError(
            f"{fsa}: missing demand values were found in the observed series."
        )

    return frame


def build_future_exog(
    forecast_origin: pd.Timestamp,
    horizon: int = 24,
) -> pd.DataFrame:
    """Build exogenous regressors for t+1 ... t+horizon."""
    future_times = pd.date_range(
        start=pd.Timestamp(forecast_origin) + pd.Timedelta(hours=1),
        periods=int(horizon),
        freq="h",
    )
    return build_calendar_exog(future_times)
