"""Validation checks for the common Modeling Foundation."""

from __future__ import annotations

import pandas as pd

from .splits import (
    TemporalFold,
    forecast_origin_end,
    slice_forecast_origin_fold,
)


def validate_temporal_folds(
    frame: pd.DataFrame,
    folds: list[TemporalFold],
    time_column: str,
    group_column: str = "fsa",
    max_horizon_hours: int = 24,
) -> pd.DataFrame:
    """
    Validate horizon-safe temporal separation and FSA coverage.
    """

    rows = []

    for fold in folds:

        train, evaluation = (
            slice_forecast_origin_fold(
                frame,
                fold,
                time_column=time_column,
                max_horizon_hours=max_horizon_hours,
            )
        )

        train_origin_end = forecast_origin_end(
            fold.train_end,
            max_horizon_hours,
        )

        evaluation_origin_end = (
            forecast_origin_end(
                fold.evaluation_end,
                max_horizon_hours,
            )
        )

        checks = {
            "non_empty_train": (
                len(train) > 0
            ),
            "non_empty_evaluation": (
                len(evaluation) > 0
            ),
            "train_precedes_evaluation": (
                train[time_column].max()
                < evaluation[time_column].min()
            ),
            "same_fsa_coverage": (
                set(
                    train[
                        group_column
                    ].dropna().unique()
                )
                == set(
                    evaluation[
                        group_column
                    ].dropna().unique()
                )
            ),
            "no_temporal_overlap": (
                train[time_column].max()
                < evaluation[time_column].min()
            ),
            "train_origin_is_horizon_safe": (
                train[time_column].max()
                <= train_origin_end
            ),
            "evaluation_origin_is_horizon_safe": (
                evaluation[time_column].max()
                <= evaluation_origin_end
            ),
        }

        for check, passed in checks.items():

            rows.append(
                {
                    "fold": fold.name,
                    "role": fold.evaluation_role,
                    "check": check,
                    "status": (
                        "PASS"
                        if passed
                        else "FAIL"
                    ),
                }
            )

    return pd.DataFrame(rows)


def validate_forecast_target_horizon_safety(
    forecast_targets: pd.DataFrame,
    folds: list[TemporalFold],
    horizons: int,
    target_prefix: str = "target_h",
    time_column: str = "forecast_origin",
) -> pd.DataFrame:
    """
    Validate that every model-development forecast origin remains
    inside its own training/evaluation target period.

    This prevents target_h24 from crossing a fold boundary.
    """

    rows = []

    target_columns = [
        f"{target_prefix}{horizon:02d}"
        for horizon in range(
            1,
            horizons + 1,
        )
    ]

    missing_columns = [
        column
        for column in target_columns
        if column not in forecast_targets.columns
    ]

    rows.append(
        {
            "fold": "global",
            "role": "forecast_target",
            "check": (
                "all_forecasting_target_columns_available"
            ),
            "status": (
                "PASS"
                if not missing_columns
                else "FAIL"
            ),
            "details": (
                ""
                if not missing_columns
                else " | ".join(
                    missing_columns
                )
            ),
        }
    )

    if missing_columns:
        return pd.DataFrame(rows)

    for fold in folds:

        train, evaluation = (
            slice_forecast_origin_fold(
                forecast_targets,
                fold,
                time_column=time_column,
                max_horizon_hours=horizons,
            )
        )

        checks = [
            (
                "train_horizon_complete",
                train[target_columns]
                .notna()
                .all(axis=None),
            ),
            (
                "evaluation_horizon_complete",
                evaluation[target_columns]
                .notna()
                .all(axis=None),
            ),
            (
                "train_h24_within_training_period",
                (
                    train[time_column]
                    + pd.Timedelta(
                        hours=horizons
                    )
                ).max()
                <= fold.train_end,
            ),
            (
                "evaluation_h24_within_evaluation_period",
                (
                    evaluation[time_column]
                    + pd.Timedelta(
                        hours=horizons
                    )
                ).max()
                <= fold.evaluation_end,
            ),
        ]

        for check, passed in checks:

            rows.append(
                {
                    "fold": fold.name,
                    "role": fold.evaluation_role,
                    "check": check,
                    "status": (
                        "PASS"
                        if bool(passed)
                        else "FAIL"
                    ),
                    "details": "",
                }
            )

    return pd.DataFrame(rows)


def validate_holdout_policy(
    config: dict,
) -> pd.DataFrame:
    """
    Confirm that the final test period occurs after all validation periods.
    """

    validation_folds = (
        config["modeling"]["splits"][
            "validation_folds"
        ]
    )

    holdout = (
        config["modeling"]["splits"][
            "final_holdout"
        ]
    )

    test_start = pd.Timestamp(
        holdout["test_start"]
    )

    latest_validation_end = max(
        pd.Timestamp(
            item["validation_end"]
        )
        for item in validation_folds
    )

    passed = (
        latest_validation_end
        < test_start
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "final_holdout_after_all_validation_periods"
                ),
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                "latest_validation_end": (
                    latest_validation_end
                ),
                "test_start": test_start,
            }
        ]
    )


def validate_preprocessing_policy(
    config: dict,
) -> pd.DataFrame:
    """
    Confirm the configured anti-leakage preprocessing policy.
    """

    fit_scope = (
        config["modeling"][
            "preprocessing"
        ]["fit_scope"]
    )

    return pd.DataFrame(
        [
            {
                "check": (
                    "preprocessing_fit_scope_training_only"
                ),
                "status": (
                    "PASS"
                    if fit_scope
                    == "training_only"
                    else "FAIL"
                ),
                "configured_value": (
                    fit_scope
                ),
            }
        ]
    )


def validate_model_selection_policy(
    config: dict,
) -> pd.DataFrame:
    """
    Validate common model-selection and classification policies.
    """

    forecasting = (
        config["modeling"]["forecasting"]
    )

    peak_risk = (
        config["modeling"]["peak_risk"]
    )

    probability_threshold = float(
        peak_risk["probability_threshold"]
    )

    checks = {
        "forecasting_selection_metric_is_primary": (
            forecasting[
                "selection_metric"
            ]
            in forecasting[
                "primary_metrics"
            ]
        ),
        "peak_selection_metric_is_primary": (
            peak_risk[
                "selection_metric"
            ]
            in peak_risk[
                "primary_metrics"
            ]
        ),
        "classification_probability_threshold_valid": (
            0.0
            < probability_threshold
            < 1.0
        ),
        "forecast_and_peak_horizons_match": (
            int(
                forecasting[
                    "horizons"
                ]
            )
            == int(
                peak_risk[
                    "horizons"
                ]
            )
        ),
    }

    return pd.DataFrame(
        [
            {
                "check": check,
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
            for check, passed
            in checks.items()
        ]
    )


def validate_fold_fsa_coverage(
    frame: pd.DataFrame,
    folds: list[TemporalFold],
    time_column: str,
    group_column: str = "fsa",
    max_horizon_hours: int = 24,
) -> pd.DataFrame:
    """
    Validate row coverage for every FSA within each horizon-safe fold.
    """

    rows = []

    for fold in folds:

        train, evaluation = (
            slice_forecast_origin_fold(
                frame,
                fold,
                time_column=time_column,
                max_horizon_hours=max_horizon_hours,
            )
        )

        fsas = sorted(
            set(
                train[
                    group_column
                ].dropna().unique()
            )
            | set(
                evaluation[
                    group_column
                ].dropna().unique()
            )
        )

        for fsa in fsas:

            train_rows = int(
                (
                    train[group_column]
                    == fsa
                ).sum()
            )

            evaluation_rows = int(
                (
                    evaluation[group_column]
                    == fsa
                ).sum()
            )

            rows.append(
                {
                    "fold": fold.name,
                    "role": (
                        fold.evaluation_role
                    ),
                    "fsa": fsa,
                    "train_rows": (
                        train_rows
                    ),
                    "evaluation_rows": (
                        evaluation_rows
                    ),
                    "status": (
                        "PASS"
                        if (
                            train_rows > 0
                            and evaluation_rows > 0
                        )
                        else "FAIL"
                    ),
                }
            )

    return pd.DataFrame(rows)
