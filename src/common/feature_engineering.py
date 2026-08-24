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
HOT_THRESHOLD_C = 28.0
COLD_THRESHOLD_C = -10.0


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


def add_temperature_episode_features(
    df: pd.DataFrame,
    hot_threshold_c: float = HOT_THRESHOLD_C,
    cold_threshold_c: float = COLD_THRESHOLD_C,
    temperature_column: str = "temperature_c",
) -> pd.DataFrame:
    """Add running hour-counts of the current hot/cold spell, per FSA.

    ``heat_episode_hours``/``cold_episode_hours`` count consecutive hours at or above
    ``hot_threshold_c`` / at or below ``cold_threshold_c``, resetting to 0 when the streak
    breaks (including at FSA boundaries and missing readings).
    """
    df = df.sort_values(["fsa", "timestamp_local"]).copy()
    new_fsa = df["fsa"] != df["fsa"].shift()

    for column, condition in [
        ("heat_episode_hours", df[temperature_column] >= hot_threshold_c),
        ("cold_episode_hours", df[temperature_column] <= cold_threshold_c),
    ]:
        condition = condition.fillna(False)
        streak_id = (~condition | new_fsa).cumsum()
        df[column] = condition.astype(int).groupby(streak_id).cumsum()

    return df
