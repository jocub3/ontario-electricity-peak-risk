"""Feature and target engineering for the multi-output LightGBM regressor.

Adds consumption lags/rolling stats and the multi-output targets
(`target_h1` .. `target_h24`). Cyclical calendar features already come from
the join in ``data_loading.py``, so this module does not add them again.
"""

from __future__ import annotations

import pandas as pd

CONSUMPTION_COLUMN = "electricity_consumption"
LAG_HOURS = (1, 2, 3, 24, 48, 168)
ROLLING_WINDOWS = (24, 168)
HORIZONS = tuple(range(1, 25))


def add_lag_features(
    df: pd.DataFrame,
    lag_hours: tuple[int, ...] = LAG_HOURS,
    consumption_column: str = CONSUMPTION_COLUMN,
) -> pd.DataFrame:
    """Add trailing consumption lags (t-1h, t-24h, ...), per FSA."""
    df = df.sort_values(["fsa", "timestamp_local"]).copy()
    grouped = df.groupby("fsa")[consumption_column]
    for lag in lag_hours:
        df[f"consumption_lag_{lag}h"] = grouped.shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    consumption_column: str = CONSUMPTION_COLUMN,
) -> pd.DataFrame:
    """Add trailing rolling mean/std of consumption, window ending at each row's own hour, per FSA."""
    df = df.sort_values(["fsa", "timestamp_local"]).copy()
    grouped = df.groupby("fsa")[consumption_column]
    for window in windows:
        df[f"consumption_roll_mean_{window}h"] = grouped.transform(
            lambda s, w=window: s.rolling(w, min_periods=w).mean()
        )
        df[f"consumption_roll_std_{window}h"] = grouped.transform(
            lambda s, w=window: s.rolling(w, min_periods=w).std()
        )
    return df


def add_multi_output_targets(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    consumption_column: str = CONSUMPTION_COLUMN,
) -> pd.DataFrame:
    """Add one target column per forecast horizon (`target_h{n}`), per FSA.

    Each is the actual consumption n hours ahead, via a forward shift per
    FSA. The last ``max(horizons)`` rows of each FSA end up NaN, with no
    future data left to shift in.
    """
    df = df.sort_values(["fsa", "timestamp_local"]).copy()
    grouped = df.groupby("fsa")[consumption_column]
    for horizon in horizons:
        df[f"target_h{horizon}"] = grouped.shift(-horizon)
    return df
