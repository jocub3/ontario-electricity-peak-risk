
"""Generate a reproducible Data Quality Report for all project datasets.

Outputs:
- docs/Data_Quality_Report.md
- reports/data_quality_dataset_summary.csv
- reports/data_quality_column_summary.csv
- reports/data_quality_key_summary.csv
- reports/data_quality_temporal_summary.csv
- reports/data_quality_numeric_summary.csv
- reports/data_quality_category_summary.csv
- reports/data_quality_join_readiness.csv

The report describes issues; it does not silently delete, impute, or correct data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.ontario_peak_risk.utils.io import (
    ensure_output_directories,
    load_project_datasets,
)


DATASET_CONFIG = {
    "consumption": {
        "timestamp": "TIMESTAMP",
        "entity": ["FSA"],
        "exact_key": [
            "FSA",
            "TIMESTAMP",
            "CUSTOMER_TYPE",
            "PRICE_PLAN",
        ],
        "expected_frequency": "h",
    },
    "weather": {
        "timestamp": "Date/Time (LST)",
        "entity": ["Climate ID", "Station Name"],
        "exact_key": ["Climate ID", "Date/Time (LST)"],
        "expected_frequency": "h",
    },
    "calendar": {
        "timestamp": "timestamp_local",
        "entity": [],
        "exact_key": ["timestamp_local"],
        "expected_frequency": "h",
    },
}


def _dataset_summary(
    dataset_name: str,
    frame: pd.DataFrame,
    timestamp_column: str,
) -> dict[str, Any]:
    timestamp = pd.to_datetime(frame[timestamp_column], errors="coerce")
    return {
        "dataset": dataset_name,
        "rows": len(frame),
        "columns": frame.shape[1] - int("_SOURCE_PATH" in frame.columns),
        "memory_mb": round(frame.memory_usage(deep=True).sum() / 1024**2, 3),
        "exact_duplicate_rows": int(frame.drop(columns=["_SOURCE_PATH"], errors="ignore").duplicated().sum()),
        "minimum_timestamp": timestamp.min(),
        "maximum_timestamp": timestamp.max(),
        "missing_timestamp_rows": int(timestamp.isna().sum()),
        "source_file_count": int(frame["_SOURCE_PATH"].nunique())
        if "_SOURCE_PATH" in frame.columns
        else pd.NA,
    }


def _column_summary(
    dataset_name: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column in frame.columns:
        if column == "_SOURCE_PATH":
            continue

        series = frame[column]
        missing_count = int(series.isna().sum())
        row_count = len(series)

        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "dtype": str(series.dtype),
                "rows": row_count,
                "non_null_count": int(series.notna().sum()),
                "missing_count": missing_count,
                "missing_pct": round(missing_count / row_count * 100, 4)
                if row_count
                else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "constant_column": bool(series.nunique(dropna=True) <= 1),
            }
        )

    return pd.DataFrame(rows)


def _key_summary(
    dataset_name: str,
    frame: pd.DataFrame,
    key_columns: list[str],
) -> dict[str, Any]:
    available = [column for column in key_columns if column in frame.columns]
    missing_key_rows = int(frame[available].isna().any(axis=1).sum()) if available else len(frame)
    duplicate_key_rows = int(frame.duplicated(subset=available, keep=False).sum()) if available else pd.NA
    duplicate_key_groups = (
        int(frame.loc[frame.duplicated(subset=available, keep=False), available].drop_duplicates().shape[0])
        if available
        else pd.NA
    )

    return {
        "dataset": dataset_name,
        "expected_key": " + ".join(key_columns),
        "available_key_columns": " + ".join(available),
        "missing_key_rows": missing_key_rows,
        "duplicate_key_rows": duplicate_key_rows,
        "duplicate_key_groups": duplicate_key_groups,
        "key_is_unique": bool(duplicate_key_rows == 0) if available else False,
    }


def _expected_hour_index(
    minimum: pd.Timestamp,
    maximum: pd.Timestamp,
) -> pd.DatetimeIndex:
    if pd.isna(minimum) or pd.isna(maximum):
        return pd.DatetimeIndex([])
    return pd.date_range(minimum.floor("h"), maximum.floor("h"), freq="h")


def _temporal_summary(
    dataset_name: str,
    frame: pd.DataFrame,
    timestamp_column: str,
    entity_columns: list[str],
) -> pd.DataFrame:
    working = frame.copy()
    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column], errors="coerce"
    )
    working = working.dropna(subset=[timestamp_column])

    rows: list[dict[str, Any]] = []

    if entity_columns:
        grouped = working.groupby(entity_columns, dropna=False, sort=True)
    else:
        grouped = [("ALL", working)]

    for entity, group in grouped:
        unique_timestamp = (
            group[timestamp_column]
            .drop_duplicates()
            .sort_values()
        )
        minimum = unique_timestamp.min()
        maximum = unique_timestamp.max()
        expected = _expected_hour_index(minimum, maximum)
        observed_index = pd.DatetimeIndex(unique_timestamp)
        missing = expected.difference(observed_index)
        duplicate_timestamps = int(
            group.duplicated(subset=[timestamp_column], keep=False).sum()
        )

        if entity_columns:
            if not isinstance(entity, tuple):
                entity = (entity,)
            entity_text = " | ".join(
                f"{column}={value}"
                for column, value in zip(entity_columns, entity)
            )
        else:
            entity_text = "ALL"

        daily_counts = (
            group.groupby(group[timestamp_column].dt.date)[timestamp_column]
            .nunique()
        )

        rows.append(
            {
                "dataset": dataset_name,
                "entity": entity_text,
                "minimum_timestamp": minimum,
                "maximum_timestamp": maximum,
                "observed_unique_hours": len(observed_index),
                "expected_continuous_hours": len(expected),
                "missing_hour_count": len(missing),
                "duplicate_timestamp_rows": duplicate_timestamps,
                "days_with_less_than_24_unique_hours": int((daily_counts < 24).sum()),
                "days_with_more_than_24_unique_hours": int((daily_counts > 24).sum()),
                "minimum_hours_in_a_day": int(daily_counts.min()) if not daily_counts.empty else pd.NA,
                "maximum_hours_in_a_day": int(daily_counts.max()) if not daily_counts.empty else pd.NA,
                "first_missing_hour_examples": " | ".join(
                    missing[:5].strftime("%Y-%m-%d %H:%M:%S").tolist()
                ),
            }
        )

    return pd.DataFrame(rows)


def _numeric_summary(
    dataset_name: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    numeric = frame.select_dtypes(include=[np.number])

    for column in numeric.columns:
        series = numeric[column].dropna()

        if series.empty:
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "count": 0,
                    "minimum": pd.NA,
                    "p01": pd.NA,
                    "p25": pd.NA,
                    "median": pd.NA,
                    "mean": pd.NA,
                    "p75": pd.NA,
                    "p99": pd.NA,
                    "maximum": pd.NA,
                    "standard_deviation": pd.NA,
                    "iqr_outlier_count": 0,
                    "iqr_outlier_pct": 0.0,
                }
            )
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())

        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "count": len(series),
                "minimum": series.min(),
                "p01": series.quantile(0.01),
                "p25": q1,
                "median": series.median(),
                "mean": series.mean(),
                "p75": q3,
                "p99": series.quantile(0.99),
                "maximum": series.max(),
                "standard_deviation": series.std(),
                "iqr_outlier_count": outlier_count,
                "iqr_outlier_pct": round(outlier_count / len(series) * 100, 4),
            }
        )

    return pd.DataFrame(rows)


def _category_summary(
    dataset_name: str,
    frame: pd.DataFrame,
    maximum_unique: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column in frame.columns:
        if column == "_SOURCE_PATH":
            continue

        series = frame[column]
        unique = series.nunique(dropna=True)

        if unique == 0 or unique > maximum_unique:
            continue

        counts = (
            series.fillna("<MISSING>")
            .astype(str)
            .value_counts(dropna=False)
        )

        for value, count in counts.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "value": value,
                    "count": int(count),
                    "pct": round(count / len(series) * 100, 4)
                    if len(series)
                    else 0.0,
                }
            )

    return pd.DataFrame(rows)


def _join_readiness(
    consumption: pd.DataFrame,
    weather: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Compare timestamp coverage without performing the final analytical join."""
    consumption_time = pd.to_datetime(
        consumption["TIMESTAMP"], errors="coerce"
    ).dropna()
    weather_time = pd.to_datetime(
        weather["Date/Time (LST)"], errors="coerce"
    ).dropna()
    calendar_time = pd.to_datetime(
        calendar["timestamp_local"], errors="coerce"
    ).dropna()

    calendar_set = pd.DatetimeIndex(calendar_time.drop_duplicates())

    rows: list[dict[str, Any]] = []

    for fsa, group in consumption.assign(_TIME=consumption_time).groupby("FSA"):
        observed = pd.DatetimeIndex(group["_TIME"].dropna().drop_duplicates())
        rows.append(
            {
                "dataset": "consumption",
                "entity": f"FSA={fsa}",
                "unique_timestamps": len(observed),
                "timestamps_missing_from_calendar": len(observed.difference(calendar_set)),
                "calendar_timestamps_without_entity_data": len(calendar_set.difference(observed)),
            }
        )

    weather_working = weather.copy()
    weather_working["_TIME"] = pd.to_datetime(
        weather_working["Date/Time (LST)"], errors="coerce"
    )

    for (climate_id, station), group in weather_working.groupby(
        ["Climate ID", "Station Name"], dropna=False
    ):
        observed = pd.DatetimeIndex(group["_TIME"].dropna().drop_duplicates())
        rows.append(
            {
                "dataset": "weather",
                "entity": f"Climate ID={climate_id} | Station={station}",
                "unique_timestamps": len(observed),
                "timestamps_missing_from_calendar": len(observed.difference(calendar_set)),
                "calendar_timestamps_without_entity_data": len(calendar_set.difference(observed)),
            }
        )

    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)


def _write_markdown(
    output_path: Path,
    start_year: int,
    end_year: int,
    dataset_summary: pd.DataFrame,
    key_summary: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    column_summary: pd.DataFrame,
    join_readiness: pd.DataFrame,
) -> None:
    high_missing = (
        column_summary.loc[column_summary["missing_pct"] >= 10]
        .sort_values(["dataset", "missing_pct"], ascending=[True, False])
    )
    constant_columns = column_summary.loc[column_summary["constant_column"]]

    sections = [
        "# Data Quality Report",
        "",
        f"**Analysis period:** {start_year}-01-01 through {end_year}-12-31.",
        "",
        "This report profiles the combined consumption, weather, and calendar "
        "datasets. It identifies potential issues but does not automatically "
        "delete, impute, aggregate, or correct records.",
        "",
        "## Dataset overview",
        "",
        _markdown_table(dataset_summary),
        "",
        "## Key integrity",
        "",
        _markdown_table(key_summary),
        "",
        "### Interpretation note",
        "",
        "Consumption records are expected to repeat by FSA and timestamp because "
        "the source grain also includes customer type and price plan. Therefore, "
        "the detailed consumption key includes those fields.",
        "",
        "## Temporal completeness",
        "",
        _markdown_table(temporal_summary),
        "",
        "## Join readiness against the calendar",
        "",
        _markdown_table(join_readiness),
        "",
        "## Columns with at least 10% missing values",
        "",
        _markdown_table(high_missing),
        "",
        "Missingness in derived weather measures such as humidex or wind chill "
        "may be condition-dependent and should not automatically be interpreted "
        "as a data error. Source flags must be reviewed before imputation.",
        "",
        "## Constant or single-valued columns",
        "",
        _markdown_table(constant_columns),
        "",
        "## Required decisions before the Master Dataset",
        "",
        "1. Confirm the final timestamp convention used to join consumption, "
        "weather, and calendar.",
        "2. Define the aggregation rule from detailed IESO records to one demand "
        "value per FSA and hour.",
        "3. Review ECCC flag-code meanings and decide whether flags become quality "
        "filters, model features, or documentation fields.",
        "4. Establish variable-specific missing-value rules; do not apply one "
        "generic imputation method to all weather fields.",
        "5. Investigate duplicate keys, missing hours, and abnormal values before "
        "building modeling features.",
        "",
        "Detailed machine-readable results are available in the CSV files under "
        "`reports/`.",
        "",
    ]

    output_path.write_text("\n".join(sections), encoding="utf-8")


def build_data_quality_report(
    processed_directory: str | Path,
    project_root: str | Path,
    *,
    start_year: int = 2021,
    end_year: int = 2025,
) -> dict[str, pd.DataFrame]:
    datasets = load_project_datasets(
        processed_directory,
        start_year=start_year,
        end_year=end_year,
    )

    dataset_frames = {
        "consumption": datasets.consumption,
        "weather": datasets.weather,
        "calendar": datasets.calendar,
    }

    dataset_summary = pd.DataFrame(
        [
            _dataset_summary(
                name,
                frame,
                DATASET_CONFIG[name]["timestamp"],
            )
            for name, frame in dataset_frames.items()
        ]
    )

    column_summary = pd.concat(
        [
            _column_summary(name, frame)
            for name, frame in dataset_frames.items()
        ],
        ignore_index=True,
    )

    key_summary = pd.DataFrame(
        [
            _key_summary(
                name,
                frame,
                DATASET_CONFIG[name]["exact_key"],
            )
            for name, frame in dataset_frames.items()
        ]
    )

    temporal_summary = pd.concat(
        [
            _temporal_summary(
                name,
                frame,
                DATASET_CONFIG[name]["timestamp"],
                DATASET_CONFIG[name]["entity"],
            )
            for name, frame in dataset_frames.items()
        ],
        ignore_index=True,
    )

    numeric_summary = pd.concat(
        [
            _numeric_summary(name, frame)
            for name, frame in dataset_frames.items()
        ],
        ignore_index=True,
    )

    category_summary = pd.concat(
        [
            _category_summary(name, frame)
            for name, frame in dataset_frames.items()
        ],
        ignore_index=True,
    )

    join_readiness = _join_readiness(
        datasets.consumption,
        datasets.weather,
        datasets.calendar,
    )

    docs_directory, reports_directory = ensure_output_directories(project_root)

    outputs = {
        "dataset_summary": dataset_summary,
        "column_summary": column_summary,
        "key_summary": key_summary,
        "temporal_summary": temporal_summary,
        "numeric_summary": numeric_summary,
        "category_summary": category_summary,
        "join_readiness": join_readiness,
    }

    for name, frame in outputs.items():
        frame.to_csv(
            reports_directory / f"data_quality_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    markdown_path = docs_directory / "Data_Quality_Report.md"
    _write_markdown(
        markdown_path,
        start_year,
        end_year,
        dataset_summary,
        key_summary,
        temporal_summary,
        column_summary,
        join_readiness,
    )

    print(f"Data Quality Report: {markdown_path.resolve()}")
    for name in outputs:
        print(
            f"{name}: "
            f"{(reports_directory / f'data_quality_{name}.csv').resolve()}"
        )

    return outputs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Capstone Data Quality Report."
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing processed consumption, weather, and calendar files.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root containing docs/ and reports/.",
    )
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    build_data_quality_report(
        processed_directory=args.processed_dir,
        project_root=args.project_root,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()
