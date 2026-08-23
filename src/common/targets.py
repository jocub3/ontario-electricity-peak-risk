"""Origin to horizon pivot: builds the forecast targets (actual consumption h hours ahead)
in long form (one row per FSA/origin/horizon, used by the classifier) and wide form
(one row per FSA/origin with 24 horizon columns, used by the multi-output regressor).

Each FSA is a complete, gap-free hourly grid, confirmed during cleaning, so "h hours
later" is exactly "h rows later" within that FSA's own series, shifting per FSA group is
enough, no timestamp lookup or join needed. Origins near the end of each FSA's series do
not have a full 24 hours of future data available and are dropped.
"""

import pandas as pd

HORIZONS = range(1, 25)


def build_long_targets(df: pd.DataFrame, value_col: str = "TOTAL_CONSUMPTION") -> pd.DataFrame:
    """One row per (FSA, origin_timestamp, horizon): actual consumption h hours later.

    Keeps only origins where every horizon 1..24 has a real actual value, matching
    build_wide_targets() exactly - an origin near the end of an FSA's series with only
    some horizons available gets dropped entirely, not kept with a partial set of rows.
    """
    df = df.sort_values(["FSA", "timestamp_local"]).reset_index(drop=True)
    grouped = df.groupby("FSA", observed=True)[value_col]

    shifts = {h: grouped.shift(-h) for h in HORIZONS}
    valid = shifts[max(HORIZONS)].notna()

    frames = []
    for h in HORIZONS:
        frames.append(
            pd.DataFrame(
                {
                    "FSA": df["FSA"],
                    "origin_timestamp": df["timestamp_local"],
                    "horizon": h,
                    "forecast_timestamp": df["timestamp_local"] + pd.Timedelta(hours=h),
                    "actual": shifts[h].values,
                }
            )[valid.values]
        )

    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def build_wide_targets(df: pd.DataFrame, value_col: str = "TOTAL_CONSUMPTION") -> pd.DataFrame:
    """One row per (FSA, origin_timestamp), with columns actual_h1..actual_h24."""
    df = df.sort_values(["FSA", "timestamp_local"]).reset_index(drop=True)

    wide = pd.DataFrame({"FSA": df["FSA"], "origin_timestamp": df["timestamp_local"]})
    for h in HORIZONS:
        wide[f"actual_h{h}"] = df.groupby("FSA", observed=True)[value_col].shift(-h).values

    return wide.dropna().reset_index(drop=True)
