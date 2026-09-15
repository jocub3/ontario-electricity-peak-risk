"""Build the 24-hour electricity-demand forecasting target matrix."""

from __future__ import annotations

import pandas as pd


def build_forecasting_target_matrix(
    frame: pd.DataFrame,
    target_column: str,
    horizon_hours: int = 24,
    target_prefix: str = "target_h",
) -> pd.DataFrame:
    """
    Build a leakage-safe label matrix for h+1 ... h+H.

    The method uses explicit FSA + future timestamp lookup rather than
    relying only on row position. This prevents target misalignment if
    an hourly timestamp is unexpectedly missing.
    """
    lookup = (
        frame[["fsa", "timestamp", target_column]]
        .drop_duplicates(["fsa", "timestamp"])
        .set_index(["fsa", "timestamp"])[target_column]
    )

    output = frame[["fsa", "timestamp"]].copy()
    output = output.rename(
        columns={"timestamp": "forecast_origin"}
    )

    for horizon in range(1, horizon_hours + 1):
        target_name = f"{target_prefix}{horizon:02d}"
        target_timestamp = (
            output["forecast_origin"]
            + pd.to_timedelta(horizon, unit="h")
        )

        keys = pd.MultiIndex.from_arrays(
            [
                output["fsa"].to_numpy(),
                target_timestamp.to_numpy(),
            ],
            names=["fsa", "timestamp"],
        )

        output[target_name] = lookup.reindex(keys).to_numpy()

    target_columns = [
        f"{target_prefix}{horizon:02d}"
        for horizon in range(1, horizon_hours + 1)
    ]

    output["available_horizons"] = (
        output[target_columns]
        .notna()
        .sum(axis=1)
        .astype("int16")
    )

    output["complete_24h_target"] = (
        output["available_horizons"] == horizon_hours
    ).astype("int8")

    return output


def forecasting_target_summary(
    target_matrix: pd.DataFrame,
    target_prefix: str = "target_h",
) -> pd.DataFrame:
    """Summarize target coverage by forecast horizon."""
    target_columns = [
        column
        for column in target_matrix.columns
        if column.startswith(target_prefix)
    ]

    rows = []

    for column in target_columns:
        rows.append(
            {
                "target": column,
                "available_count": int(
                    target_matrix[column].notna().sum()
                ),
                "missing_count": int(
                    target_matrix[column].isna().sum()
                ),
                "coverage_pct": round(
                    target_matrix[column].notna().mean() * 100,
                    6,
                ),
            }
        )

    return pd.DataFrame(rows)
