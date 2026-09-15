"""Temporal alignment checks for 24-hour forecasting targets."""

from __future__ import annotations

import pandas as pd


def build_alignment_sample(
    frame: pd.DataFrame,
    target_matrix: pd.DataFrame,
    target_column: str,
    horizons: tuple[int, ...] = (1, 2, 24),
    rows_per_fsa: int = 3,
) -> pd.DataFrame:
    """Create human-readable examples of origin-to-target alignment."""
    source = (
        frame[["fsa", "timestamp", target_column]]
        .drop_duplicates(["fsa", "timestamp"])
        .set_index(["fsa", "timestamp"])[target_column]
    )

    samples = []

    for fsa, group in target_matrix.groupby(
        "fsa",
        observed=True,
        sort=True,
    ):
        candidates = (
            group.loc[
                group["complete_24h_target"] == 1
            ]
            .head(rows_per_fsa)
        )

        for _, row in candidates.iterrows():
            for horizon in horizons:
                target_timestamp = (
                    row["forecast_origin"]
                    + pd.Timedelta(hours=horizon)
                )

                expected = source.get(
                    (fsa, target_timestamp),
                    pd.NA,
                )

                actual = row[
                    f"target_h{horizon:02d}"
                ]

                samples.append(
                    {
                        "fsa": fsa,
                        "forecast_origin": row["forecast_origin"],
                        "horizon_hours": horizon,
                        "target_timestamp": target_timestamp,
                        "target_matrix_value": actual,
                        "source_value": expected,
                        "match": (
                            pd.isna(actual)
                            and pd.isna(expected)
                        )
                        or (
                            pd.notna(actual)
                            and pd.notna(expected)
                            and actual == expected
                        ),
                    }
                )

    return pd.DataFrame(samples)
