"""Execute the complete Phase EDA workflow and assemble the EDA report."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    dataframe_to_markdown,
    load_clean_dataset,
    load_eda_config,
    prepare_output_directories,
)
from .extremes_peaks import run_extremes_peaks
from .overview import run_overview
from .spatial_calendar import run_spatial_calendar
from .target_temporal import run_target_temporal
from .weather_missing import run_weather_missing


def assemble_eda_report(
    docs_dir: Path,
    sections: list[str],
) -> Path:
    """Create a navigation-oriented consolidated EDA report."""
    output = docs_dir / "EDA_Report.md"

    content = [
        "# Exploratory Data Analysis Report",
        "",
        "This report indexes the five EDA sections. Detailed tables and figures are "
        "stored under `reports/eda/` and `reports/figures/eda/`.",
        "",
        "## Sections",
        "",
    ]

    for section in sections:
        content.append(f"- [{Path(section).stem}]({section})")

    content.extend(
        [
            "",
            "## Methodological boundaries",
            "",
            "- The clean Master Dataset is not modified.",
            "- No modeling feature engineering is performed.",
            "- Exploratory Peak labels are temporary descriptive variables only.",
            "- Missing-value and imputation decisions for modeling remain outside EDA.",
            "- Correlation does not establish causality or final feature importance.",
            "",
        ]
    )

    output.write_text("\n".join(content), encoding="utf-8")
    return output


def run_complete_eda(
    config_path: str | Path = "configs/eda.yaml",
) -> dict[str, object]:
    """Run all Phase 5 EDA sections."""
    config, project_root = load_eda_config(config_path)
    reports_dir, figures_dir, docs_dir = prepare_output_directories(
        config,
        project_root,
    )
    frame = load_clean_dataset(config, project_root)

    target = config["columns"]["target"]
    weather_columns = config["columns"]["core_weather"]
    analysis = config["analysis"]

    overview = run_overview(
        frame,
        reports_dir,
        figures_dir,
        docs_dir,
    )
    target_temporal = run_target_temporal(
        frame,
        target,
        analysis["autocorrelation_lags"],
        reports_dir,
        figures_dir,
        docs_dir,
    )
    spatial_calendar = run_spatial_calendar(
        frame,
        target,
        reports_dir,
        figures_dir,
        docs_dir,
    )
    weather_missing = run_weather_missing(
        frame,
        target,
        weather_columns,
        analysis["high_missingness_threshold_pct"],
        analysis["minimum_weather_coverage_pct"],
        reports_dir,
        figures_dir,
        docs_dir,
    )
    extremes_peaks = run_extremes_peaks(
        frame,
        target,
        analysis["exploratory_peak_percentile"],
        analysis["top_n_extreme_hours"],
        analysis["top_n_extreme_days"],
        reports_dir,
        figures_dir,
        docs_dir,
    )

    report_path = assemble_eda_report(
        docs_dir,
        [
            "05_01_Dataset_Overview_and_Coverage.md",
            "05_02_Target_and_Temporal_Analysis.md",
            "05_03_Spatial_and_Calendar_Analysis.md",
            "05_04_Weather_Correlation_and_Missingness.md",
            "05_05_Extremes_Peaks_and_Feature_Review.md",
        ],
    )

    print(f"EDA completed. Main report: {report_path.resolve()}")
    print(f"Tables: {reports_dir.resolve()}")
    print(f"Figures: {figures_dir.resolve()}")

    return {
        "dataset": frame,
        "overview": overview,
        "target_temporal": target_temporal,
        "spatial_calendar": spatial_calendar,
        "weather_missing": weather_missing,
        "extremes_peaks": extremes_peaks,
        "report_path": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Capstone EDA.")
    parser.add_argument(
        "--config",
        default="configs/eda.yaml",
        help="Path to the Phase 5 EDA YAML configuration.",
    )
    args = parser.parse_args()
    run_complete_eda(args.config)


if __name__ == "__main__":
    main()
