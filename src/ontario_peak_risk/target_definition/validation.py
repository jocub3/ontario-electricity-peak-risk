"""Validation rules for Target Definition Phase target construction."""

from __future__ import annotations

import pandas as pd


def validate_forecasting_targets(
    frame: pd.DataFrame,
    target_matrix: pd.DataFrame,
    target_column: str,
    horizon_hours: int,
) -> pd.DataFrame:
    """Validate forecasting target structure and alignment."""
    checks = []

    def add(
        name: str,
        violations: int,
        description: str,
    ) -> None:
        checks.append(
            {
                "check": name,
                "violations": int(violations),
                "status": (
                    "PASS"
                    if violations == 0
                    else "FAIL"
                ),
                "description": description,
            }
        )

    add(
        "target_origin_row_count",
        abs(len(frame) - len(target_matrix)),
        "Target matrix must preserve one forecast origin per feature row.",
    )

    add(
        "unique_forecast_origin",
        int(
            target_matrix.duplicated(
                ["fsa", "forecast_origin"],
                keep=False,
            ).sum()
        ),
        "FSA + forecast origin must remain unique.",
    )

    source = (
        frame[["fsa", "timestamp", target_column]]
        .drop_duplicates(["fsa", "timestamp"])
        .set_index(["fsa", "timestamp"])[target_column]
    )

    for horizon in range(1, horizon_hours + 1):
        column = f"target_h{horizon:02d}"

        target_timestamp = (
            target_matrix["forecast_origin"]
            + pd.to_timedelta(horizon, unit="h")
        )

        keys = pd.MultiIndex.from_arrays(
            [
                target_matrix["fsa"].to_numpy(),
                target_timestamp.to_numpy(),
            ],
            names=["fsa", "timestamp"],
        )

        expected = pd.Series(
            source.reindex(keys).to_numpy(),
            index=target_matrix.index,
        )
        actual = target_matrix[column]

        mismatch = ~(
            actual.eq(expected)
            | (
                actual.isna()
                & expected.isna()
            )
        )

        add(
            f"alignment_h{horizon:02d}",
            int(mismatch.sum()),
            f"h+{horizon} target must equal observed consumption exactly {horizon} hour(s) after forecast origin.",
        )

    return pd.DataFrame(checks)


def validate_peak_diagnostics(
    labels: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the walk-forward Peak diagnostic process."""
    checks = []

    def add(
        name: str,
        violations: int,
        description: str,
    ) -> None:
        checks.append(
            {
                "check": name,
                "violations": int(violations),
                "status": (
                    "PASS"
                    if violations == 0
                    else "FAIL"
                ),
                "description": description,
            }
        )

    if labels.empty:
        add(
            "diagnostic_labels_created",
            1,
            "At least one leakage-safe evaluation year must be available.",
        )
        return pd.DataFrame(checks)

    add(
        "unique_peak_label_key",
        int(
            labels.duplicated(
                ["fsa", "timestamp"],
                keep=False,
            ).sum()
        ),
        "Diagnostic Peak labels must be unique by FSA + timestamp.",
    )

    add(
        "peak_label_binary",
        int(
            (
                labels["peak_risk_target"]
                .dropna()
                .isin([0, 1])
                == False
            ).sum()
        ),
        "Peak-Risk labels must be binary when a threshold is available.",
    )

    add(
        "peak_threshold_availability_consistency",
        int(
            (
                labels["peak_threshold_available"].eq(1)
                & labels["peak_risk_target"].isna()
            ).sum()
        ),
        "Available thresholds must produce a diagnostic binary label.",
    )

    if not thresholds.empty:
        leakage_violations = (
            thresholds["training_year_end"]
            >= thresholds["evaluation_year"]
        ).sum()

        add(
            "walk_forward_threshold_no_future_year",
            int(leakage_violations),
            "Each diagnostic threshold must be fitted using years strictly before its evaluation year.",
        )

    return pd.DataFrame(checks)
