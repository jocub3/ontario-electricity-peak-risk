"""Execute Modeling Foundation Phase validation and documentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import (
    ensure_modeling_directories,
    load_feature_dataset,
    load_forecast_targets,
    load_modeling_config,
    save_csv,
)

from .metrics import metric_dictionary

from .peak_policy import (
    peak_horizon_policy,
)

from .reporting import (
    write_modeling_design,
    write_modeling_summary,
)

from .splits import (
    build_final_holdout_fold,
    build_validation_folds,
    split_summary,
)

from .validation import (
    validate_fold_fsa_coverage,
    validate_forecast_target_horizon_safety,
    validate_holdout_policy,
    validate_model_selection_policy,
    validate_preprocessing_policy,
    validate_temporal_folds,
)


def run_modeling_foundation(
    config_path: str | Path = (
        "configs/modeling_foundation.yaml"
    ),
) -> dict[str, object]:
    """
    Run all shared Modeling Foundation checks and reports.
    """

    config, project_root = (
        load_modeling_config(
            config_path
        )
    )

    reports_dir, docs_dir, outputs_dir = (
        ensure_modeling_directories(
            config,
            project_root,
        )
    )

    feature_dataset = (
        load_feature_dataset(
            config,
            project_root,
        )
    )

    forecast_targets = (
        load_forecast_targets(
            config,
            project_root,
        )
    )

    group_column = (
        config["modeling"]["time"][
            "group_column"
        ]
    )

    forecasting_config = (
        config["modeling"][
            "forecasting"
        ]
    )

    peak_config = (
        config["modeling"][
            "peak_risk"
        ]
    )

    horizons = int(
        forecasting_config[
            "horizons"
        ]
    )

    validation_folds = (
        build_validation_folds(
            config
        )
    )

    final_holdout = (
        build_final_holdout_fold(
            config
        )
    )

    all_folds = [
        *validation_folds,
        final_holdout,
    ]

    # ---------------------------------------------------------
    # Horizon-safe common modeling splits
    # ---------------------------------------------------------

    splits_report = split_summary(
        forecast_targets,
        all_folds,
        time_column="forecast_origin",
        group_column=group_column,
        max_horizon_hours=horizons,
    )

    # ---------------------------------------------------------
    # FSA coverage by fold
    # ---------------------------------------------------------

    fsa_coverage = (
        validate_fold_fsa_coverage(
            forecast_targets,
            all_folds,
            time_column="forecast_origin",
            group_column=group_column,
            max_horizon_hours=horizons,
        )
    )

    # ---------------------------------------------------------
    # Official metrics
    # ---------------------------------------------------------

    metrics_report = (
        metric_dictionary()
    )

    # ---------------------------------------------------------
    # Peak-Risk multi-horizon policy
    # ---------------------------------------------------------

    peak_horizon_report = (
        peak_horizon_policy(
            horizons=int(
                peak_config[
                    "horizons"
                ]
            ),
            target_prefix=(
                peak_config[
                    "target_prefix"
                ]
            ),
        )
    )

    # ---------------------------------------------------------
    # Framework validation
    # ---------------------------------------------------------

    temporal_validation = (
        validate_temporal_folds(
            forecast_targets,
            all_folds,
            time_column=(
                "forecast_origin"
            ),
            group_column=(
                group_column
            ),
            max_horizon_hours=(
                horizons
            ),
        )
    )

    horizon_validation = (
        validate_forecast_target_horizon_safety(
            forecast_targets,
            all_folds,
            horizons=horizons,
            target_prefix=(
                forecasting_config[
                    "target_prefix"
                ]
            ),
            time_column=(
                "forecast_origin"
            ),
        )
    )

    holdout_validation = (
        validate_holdout_policy(
            config
        )
    )

    preprocessing_validation = (
        validate_preprocessing_policy(
            config
        )
    )

    model_selection_validation = (
        validate_model_selection_policy(
            config
        )
    )

    framework_validation = pd.concat(
        [
            temporal_validation,
            horizon_validation,
            holdout_validation,
            preprocessing_validation,
            model_selection_validation,
        ],
        ignore_index=True,
        sort=False,
    )

    # ---------------------------------------------------------
    # Save reports
    # ---------------------------------------------------------

    save_csv(
        splits_report,
        reports_dir
        / "08_01_common_split_summary.csv",
    )

    save_csv(
        fsa_coverage,
        reports_dir
        / "08_01_fold_fsa_coverage.csv",
    )

    save_csv(
        metrics_report,
        reports_dir
        / "08_02_metric_dictionary.csv",
    )

    save_csv(
        peak_horizon_report,
        reports_dir
        / "08_06_peak_horizon_policy.csv",
    )

    save_csv(
        framework_validation,
        reports_dir
        / "08_08_framework_validation.csv",
    )

    # ---------------------------------------------------------
    # Documentation
    # ---------------------------------------------------------

    write_modeling_design(
        docs_dir,
        config,
    )

    write_modeling_summary(
        docs_dir,
        splits_report,
        metrics_report,
        framework_validation,
        fsa_coverage,
        peak_horizon_report,
    )

    # ---------------------------------------------------------
    # Final gate
    # ---------------------------------------------------------

    failed = (
        framework_validation.loc[
            framework_validation[
                "status"
            ]
            == "FAIL"
        ]
    )

    failed_fsa = (
        fsa_coverage.loc[
            fsa_coverage[
                "status"
            ]
            == "FAIL"
        ]
    )

    if (
        not failed.empty
        or not failed_fsa.empty
    ):
        raise ValueError(
            "Modeling Foundation validation failed. "
            "Review reports/modeling_foundation/."
        )

    print(
        "Modeling Foundation Phase "
        "completed successfully."
    )

    print(
        f"Feature rows: "
        f"{len(feature_dataset):,}"
    )

    print(
        f"Complete Forecasting target rows: "
        f"{len(forecast_targets):,}"
    )

    print(
        f"Maximum forecast horizon: "
        f"{horizons} hours"
    )

    print(
        f"Forecasting selection metric: "
        f"{forecasting_config['selection_metric']}"
    )

    print(
        f"Peak-Risk selection metric: "
        f"{peak_config['selection_metric']}"
    )

    print(
        f"Initial classification threshold: "
        f"{peak_config['probability_threshold']}"
    )

    print(
        f"Validation folds: "
        f"{len(validation_folds)}"
    )

    print(
        f"Final holdout: "
        f"{final_holdout.name}"
    )

    print(
        f"Reports: "
        f"{reports_dir.resolve()}"
    )

    print(
        f"Documentation: "
        f"{docs_dir.resolve()}"
    )

    return {
        "feature_dataset": (
            feature_dataset
        ),
        "forecast_targets": (
            forecast_targets
        ),
        "validation_folds": (
            validation_folds
        ),
        "final_holdout": (
            final_holdout
        ),
        "split_summary": (
            splits_report
        ),
        "fsa_coverage": (
            fsa_coverage
        ),
        "metric_dictionary": (
            metrics_report
        ),
        "peak_horizon_policy": (
            peak_horizon_report
        ),
        "framework_validation": (
            framework_validation
        ),
        "outputs_dir": (
            outputs_dir
        ),
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run Modeling Foundation Phase."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "modeling_foundation.yaml"
        ),
    )

    args = parser.parse_args()

    run_modeling_foundation(
        args.config
    )


if __name__ == "__main__":
    main()
