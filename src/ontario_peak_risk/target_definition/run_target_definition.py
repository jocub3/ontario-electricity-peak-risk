"""Execute the complete Target Definition Phase workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .alignment import build_alignment_sample
from .common import (
    ensure_directories,
    load_feature_dataset,
    load_target_config,
    save_table,
)
from .forecasting_target import (
    build_forecasting_target_matrix,
    forecasting_target_summary,
)
from .peak_risk_target import (
    build_walk_forward_peak_diagnostics,
    peak_distribution_reports,
)
from .reporting import (
    write_target_design_document,
    write_target_summary_document,
)
from .validation import (
    validate_forecasting_targets,
    validate_peak_diagnostics,
)


def run_target_definition(
    config_path: str | Path = "configs/target_definition.yaml",
) -> dict[str, object]:
    """Run the complete target-definition phase."""
    config, project_root = load_target_config(
        config_path
    )
    output_dir, reports_dir, docs_dir = (
        ensure_directories(
            config,
            project_root,
        )
    )

    frame = load_feature_dataset(
        config,
        project_root,
    )

    settings = config["target_definition"]
    forecasting = settings["forecasting"]
    peak = settings["peak_risk"]

    write_target_design_document(
        docs_dir,
        config,
    )

    forecast_matrix = (
        build_forecasting_target_matrix(
            frame,
            target_column=forecasting["target_column"],
            horizon_hours=forecasting["horizon_hours"],
            target_prefix=forecasting["target_prefix"],
        )
    )

    forecast_summary = (
        forecasting_target_summary(
            forecast_matrix,
            target_prefix=forecasting["target_prefix"],
        )
    )

    complete_forecast_matrix = (
        forecast_matrix.loc[
            forecast_matrix["complete_24h_target"] == 1
        ]
        .reset_index(drop=True)
    )

    forecast_matrix.to_parquet(
        output_dir / "forecast_target_matrix.parquet",
        index=False,
    )

    complete_forecast_matrix.to_parquet(
        output_dir / "forecast_target_complete.parquet",
        index=False,
    )

    peak_labels, peak_thresholds = (
        build_walk_forward_peak_diagnostics(
            frame,
            percentile=peak["percentile_threshold"],
            target_column=peak["target_column"],
            grouping_columns=peak["grouping_columns"],
            time_column=peak["diagnostic_time_column"],
            minimum_training_years=peak[
                "minimum_training_years"
            ],
        )
    )

    if not peak_labels.empty:
        peak_labels.to_parquet(
            output_dir
            / "peak_risk_diagnostic_labels.parquet",
            index=False,
        )

    distributions = peak_distribution_reports(
        peak_labels
    )

    alignment_sample = build_alignment_sample(
        frame,
        forecast_matrix,
        target_column=forecasting["target_column"],
    )

    forecast_validation = (
        validate_forecasting_targets(
            frame,
            forecast_matrix,
            target_column=forecasting["target_column"],
            horizon_hours=forecasting["horizon_hours"],
        )
    )

    peak_validation = validate_peak_diagnostics(
        peak_labels,
        peak_thresholds,
    )

    validation = (
        forecast_validation["status"].eq("PASS").all()
        and peak_validation["status"].eq("PASS").all()
    )

    save_table(
        forecast_summary,
        reports_dir
        / "07_01_forecasting_target_summary.csv",
    )
    save_table(
        peak_thresholds,
        reports_dir
        / "07_02_walk_forward_peak_thresholds.csv",
    )
    save_table(
        distributions["overall"],
        reports_dir
        / "07_03_peak_distribution_overall.csv",
    )
    save_table(
        distributions["by_fsa"],
        reports_dir
        / "07_03_peak_distribution_by_fsa.csv",
    )
    save_table(
        distributions["by_year"],
        reports_dir
        / "07_03_peak_distribution_by_year.csv",
    )
    save_table(
        distributions["by_season"],
        reports_dir
        / "07_03_peak_distribution_by_season.csv",
    )
    save_table(
        alignment_sample,
        reports_dir
        / "07_04_temporal_alignment_sample.csv",
    )
    save_table(
        forecast_validation,
        reports_dir
        / "07_05_forecasting_target_validation.csv",
    )
    save_table(
        peak_validation,
        reports_dir
        / "07_05_peak_target_validation.csv",
    )

    write_target_summary_document(
        docs_dir,
        forecast_summary,
        distributions["overall"],
        forecast_validation,
        peak_validation,
    )

    if not validation:
        raise ValueError(
            "One or more Target Definition validation checks failed. "
            "Review reports/target_definition/07_05_*_validation.csv."
        )

    print(
        "Target Definition Phase completed successfully."
    )
    print(
        f"Rows in feature dataset: {len(frame):,}"
    )
    print(
        "Complete 24-hour forecast origins: "
        f"{len(complete_forecast_matrix):,}"
    )
    print(
        f"Outputs: {output_dir.resolve()}"
    )
    print(
        f"Reports: {reports_dir.resolve()}"
    )
    print(
        f"Documentation: {docs_dir.resolve()}"
    )

    return {
        "feature_dataset": frame,
        "forecast_matrix": forecast_matrix,
        "complete_forecast_matrix":
            complete_forecast_matrix,
        "forecast_summary": forecast_summary,
        "peak_labels": peak_labels,
        "peak_thresholds": peak_thresholds,
        "peak_distributions": distributions,
        "alignment_sample": alignment_sample,
        "forecast_validation": forecast_validation,
        "peak_validation": peak_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Target Definition Phase."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/target_definition.yaml",
    )
    args = parser.parse_args()

    run_target_definition(
        args.config
    )


if __name__ == "__main__":
    main()
