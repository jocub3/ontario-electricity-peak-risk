"""Common chronological split definitions for all candidate models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    """One chronological train/evaluation split."""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    evaluation_role: str


def _timestamp(value: str) -> pd.Timestamp:
    """Convert a configured timestamp to pandas Timestamp."""
    return pd.Timestamp(value)


def build_validation_folds(
    config: dict,
) -> list[TemporalFold]:
    """Build expanding-window validation folds from configuration."""

    folds = []

    for item in config["modeling"]["splits"]["validation_folds"]:
        folds.append(
            TemporalFold(
                name=item["name"],
                train_start=_timestamp(item["train_start"]),
                train_end=_timestamp(item["train_end"]),
                evaluation_start=_timestamp(
                    item["validation_start"]
                ),
                evaluation_end=_timestamp(
                    item["validation_end"]
                ),
                evaluation_role="validation",
            )
        )

    return folds


def build_final_holdout_fold(
    config: dict,
) -> TemporalFold:
    """Build the untouched final test split."""

    item = config["modeling"]["splits"]["final_holdout"]

    return TemporalFold(
        name=item["name"],
        train_start=_timestamp(item["train_start"]),
        train_end=_timestamp(item["train_end"]),
        evaluation_start=_timestamp(item["test_start"]),
        evaluation_end=_timestamp(item["test_end"]),
        evaluation_role="test",
    )


def forecast_origin_end(
    period_end: pd.Timestamp,
    max_horizon_hours: int,
) -> pd.Timestamp:
    """
    Return the latest valid forecast origin for a period.

    Example:
    If a period ends at 2022-12-31 23:00 and the maximum horizon
    is 24 hours, the latest safe forecast origin is
    2022-12-30 23:00.

    This guarantees that target_h24 remains inside the same
    training/evaluation period.
    """

    if max_horizon_hours < 0:
        raise ValueError(
            "max_horizon_hours must be greater than or equal to zero."
        )

    return period_end - pd.Timedelta(
        hours=max_horizon_hours
    )


def slice_observation_fold(
    frame: pd.DataFrame,
    fold: TemporalFold,
    time_column: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the complete observation windows.

    Use this function when all observations in the configured period
    are legitimately required, for example when estimating the
    Peak-Risk percentile threshold from the training period.
    """

    train_mask = (
        (frame[time_column] >= fold.train_start)
        & (frame[time_column] <= fold.train_end)
    )

    evaluation_mask = (
        (frame[time_column] >= fold.evaluation_start)
        & (frame[time_column] <= fold.evaluation_end)
    )

    return (
        frame.loc[train_mask].copy(),
        frame.loc[evaluation_mask].copy(),
    )


def slice_forecast_origin_fold(
    frame: pd.DataFrame,
    fold: TemporalFold,
    time_column: str,
    max_horizon_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return horizon-safe forecast-origin subsets.

    Forecast origins near the end of a period are removed when their
    future targets would cross into the next temporal period.

    This is required for both:
    - 24-hour electricity-demand Forecasting;
    - 24-hour Peak-Risk prediction.
    """

    train_origin_end = forecast_origin_end(
        fold.train_end,
        max_horizon_hours,
    )

    evaluation_origin_end = forecast_origin_end(
        fold.evaluation_end,
        max_horizon_hours,
    )

    train_mask = (
        (frame[time_column] >= fold.train_start)
        & (frame[time_column] <= train_origin_end)
    )

    evaluation_mask = (
        (frame[time_column] >= fold.evaluation_start)
        & (frame[time_column] <= evaluation_origin_end)
    )

    return (
        frame.loc[train_mask].copy(),
        frame.loc[evaluation_mask].copy(),
    )


def slice_fold(
    frame: pd.DataFrame,
    fold: TemporalFold,
    time_column: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible observation-window splitter.

    New model branches should normally use
    slice_forecast_origin_fold() for multi-horizon prediction.
    """

    return slice_observation_fold(
        frame,
        fold,
        time_column=time_column,
    )


def split_summary(
    frame: pd.DataFrame,
    folds: list[TemporalFold],
    time_column: str,
    group_column: str = "fsa",
    max_horizon_hours: int = 0,
) -> pd.DataFrame:
    """
    Summarize horizon-safe model-development splits.
    """

    rows = []

    for fold in folds:

        train, evaluation = slice_forecast_origin_fold(
            frame,
            fold,
            time_column=time_column,
            max_horizon_hours=max_horizon_hours,
        )

        train_origin_end = forecast_origin_end(
            fold.train_end,
            max_horizon_hours,
        )

        evaluation_origin_end = forecast_origin_end(
            fold.evaluation_end,
            max_horizon_hours,
        )

        rows.append(
            {
                "fold": fold.name,
                "role": fold.evaluation_role,

                "configured_train_start": fold.train_start,
                "configured_train_end": fold.train_end,
                "safe_train_origin_end": train_origin_end,

                "train_rows": len(train),
                "train_first_origin": (
                    train[time_column].min()
                    if not train.empty
                    else pd.NaT
                ),
                "train_last_origin": (
                    train[time_column].max()
                    if not train.empty
                    else pd.NaT
                ),
                "train_fsas": train[group_column].nunique(),

                "configured_evaluation_start": (
                    fold.evaluation_start
                ),
                "configured_evaluation_end": (
                    fold.evaluation_end
                ),
                "safe_evaluation_origin_end": (
                    evaluation_origin_end
                ),

                "evaluation_rows": len(evaluation),
                "evaluation_first_origin": (
                    evaluation[time_column].min()
                    if not evaluation.empty
                    else pd.NaT
                ),
                "evaluation_last_origin": (
                    evaluation[time_column].max()
                    if not evaluation.empty
                    else pd.NaT
                ),
                "evaluation_fsas": (
                    evaluation[group_column].nunique()
                ),

                "max_horizon_hours": max_horizon_hours,
            }
        )

    return pd.DataFrame(rows)
