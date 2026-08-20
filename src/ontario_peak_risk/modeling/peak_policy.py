"""Shared Peak-Risk threshold and horizon policy."""

from __future__ import annotations

import pandas as pd

from src.ontario_peak_risk.target_definition.peak_risk_target import (
    PeakRiskThresholdEstimator,
)


def fit_peak_thresholds(
    training_frame: pd.DataFrame,
    percentile: float,
    target_column: str,
    grouping_columns: list[str],
) -> PeakRiskThresholdEstimator:
    """
    Fit Peak thresholds using training observations only.
    """

    return PeakRiskThresholdEstimator(
        percentile=percentile,
        target_column=target_column,
        grouping_columns=tuple(grouping_columns),
    ).fit(training_frame)


def label_peak_evaluation_data(
    estimator: PeakRiskThresholdEstimator,
    evaluation_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply frozen training-derived Peak thresholds.

    The estimator must already have been fitted using training data.
    """

    return estimator.transform(
        evaluation_frame
    )


def peak_horizon_policy(
    horizons: int = 24,
    target_prefix: str = "peak_h",
) -> pd.DataFrame:
    """
    Document the official Peak-Risk multi-horizon target structure.

    No labels are calculated here.
    """

    if horizons <= 0:
        raise ValueError(
            "horizons must be greater than zero."
        )

    rows = []

    for horizon in range(1, horizons + 1):

        rows.append(
            {
                "horizon": horizon,
                "target": (
                    f"{target_prefix}{horizon:02d}"
                ),
                "target_offset_hours": horizon,
                "definition": (
                    "Peak status of the corresponding "
                    f"FSA at forecast_origin + {horizon} hour(s)"
                ),
                "threshold_scope": (
                    "training_only_fsa_season"
                ),
            }
        )

    return pd.DataFrame(rows)


def build_peak_horizon_targets(
    labeled_frame: pd.DataFrame,
    horizons: int = 24,
    group_column: str = "fsa",
    time_column: str = "timestamp",
    label_column: str = "peak_risk_target",
    target_prefix: str = "peak_h",
    require_complete_horizon: bool = True,
) -> pd.DataFrame:
    """
    Convert hourly leakage-safe Peak labels into a multi-horizon matrix.

    IMPORTANT
    ---------
    `labeled_frame` must already have been labeled using a
    PeakRiskThresholdEstimator fitted on the appropriate training window.

    The function does not calculate thresholds and therefore cannot
    introduce a global Peak definition.

    Output:
        fsa
        forecast_origin
        peak_h01
        ...
        peak_h24
    """

    required_columns = {
        group_column,
        time_column,
        label_column,
    }

    missing_columns = sorted(
        required_columns.difference(
            labeled_frame.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Peak-labeled frame is missing required columns: "
            f"{missing_columns}"
        )

    frame = labeled_frame[
        [
            group_column,
            time_column,
            label_column,
        ]
    ].copy()

    frame[time_column] = pd.to_datetime(
        frame[time_column],
        errors="coerce",
    )

    if frame[time_column].isna().any():
        raise ValueError(
            "Peak-labeled frame contains invalid timestamps."
        )

    duplicated = frame.duplicated(
        subset=[
            group_column,
            time_column,
        ]
    )

    if duplicated.any():
        raise ValueError(
            "Peak-labeled frame contains duplicated "
            "FSA + timestamp keys."
        )

    frame = frame.sort_values(
        [
            group_column,
            time_column,
        ],
        kind="stable",
    ).reset_index(drop=True)

    origins = frame[
        [
            group_column,
            time_column,
        ]
    ].rename(
        columns={
            time_column: "forecast_origin"
        }
    )

    result = origins.copy()

    lookup = frame.set_index(
        [
            group_column,
            time_column,
        ]
    )[label_column]

    for horizon in range(
        1,
        horizons + 1,
    ):

        target_column = (
            f"{target_prefix}{horizon:02d}"
        )

        target_times = (
            result["forecast_origin"]
            + pd.to_timedelta(
                horizon,
                unit="h",
            )
        )

        keys = pd.MultiIndex.from_arrays(
            [
                result[group_column],
                target_times,
            ],
            names=[
                group_column,
                time_column,
            ],
        )

        result[target_column] = (
            lookup.reindex(keys)
            .to_numpy()
        )

    target_columns = [
        f"{target_prefix}{horizon:02d}"
        for horizon in range(
            1,
            horizons + 1,
        )
    ]

    if require_complete_horizon:
        result = result.dropna(
            subset=target_columns
        ).copy()

    for column in target_columns:
        if result[column].notna().all():
            result[column] = (
                result[column].astype("int8")
            )

    return result.reset_index(drop=True)
