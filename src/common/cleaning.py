"""Data cleaning utilities for the joined FSA-hourly dataset (see ``data_loading.py``).

Two kinds of missing weather values: short outages in ``temperature_c``/
``relative_humidity_pct`` (<0.1%, interpolated), and structural gaps in
``precipitation_mm``/``wind_speed_kmh``/``visibility_km`` (~50%, left as NaN since each FSA's
station just doesn't report them, and LightGBM handles NaN natively).
"""

from __future__ import annotations

import pandas as pd

from src.common.data_loading import FSA_STATION_MAP

INTERPOLATE_COLUMNS = ["temperature_c", "relative_humidity_pct"]


def interpolate_short_weather_gaps(df: pd.DataFrame, max_gap_hours: int = 3) -> pd.DataFrame:
    """Interpolate short weather outages in ``INTERPOLATE_COLUMNS``, per FSA.

    Fills runs up to ``max_gap_hours``; longer gaps stay NaN.
    """
    df = df.sort_values(["fsa", "timestamp_local"]).copy()
    for column in INTERPOLATE_COLUMNS:
        df[column] = df.groupby("fsa")[column].transform(
            lambda series: series.interpolate(
                method="linear", limit=max_gap_hours, limit_area="inside"
            )
        )
    return df


def find_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows sharing the same (fsa, timestamp_local) key.

    An empty result confirms the join produced exactly one row per FSA-hour.
    """
    key_columns = ["fsa", "timestamp_local"]
    duplicated = df.duplicated(subset=key_columns, keep=False)
    return df.loc[duplicated].sort_values(key_columns)


def check_station_assignment(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where ``station`` does not match the fixed ``FSA_STATION_MAP``.

    An empty result confirms every row's station matches the intended assignment.
    """
    expected_station = df["fsa"].map(FSA_STATION_MAP)
    return df.loc[df["station"] != expected_station]
