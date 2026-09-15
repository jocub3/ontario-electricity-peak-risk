"""Clean and validate the integrated Master Dataset.

Important methodological decisions
----------------------------------
- The integrated Master Dataset is preserved unchanged.
- A new `master_dataset_clean.parquet` is created for EDA.
- Rows are not deleted merely because weather values are missing.
- Numeric weather variables are not globally imputed because future observations must not influence historical training windows.
- Demand outliers are reported, not removed, because extreme demand is central to the Peak-Risk objective.
- Hard consistency violations stop the pipeline rather than being silently fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load YAML and return the configuration plus the project root."""
    resolved_config = Path(config_path).resolve()

    if not resolved_config.exists():
        raise FileNotFoundError(f"Cleaning configuration not found: {resolved_config}")

    with resolved_config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The cleaning configuration must contain a YAML mapping.")

    project_root = resolved_config.parent.parent
    return config, project_root


def resolve_project_path(project_root: Path, value: str) -> Path:
    """Resolve YAML paths consistently from the project root."""
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def as_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No records."
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)


def load_master_dataset(input_path: Path) -> pd.DataFrame:
    """Load the integrated parquet file and standardize its column labels."""
    if not input_path.exists():
        raise FileNotFoundError(f"Master Dataset not found: {input_path}")

    frame = pd.read_parquet(input_path)
    frame.columns = frame.columns.astype(str).str.strip()

    if frame.empty:
        raise ValueError("The Master Dataset contains no rows.")

    return frame


def profile_missingness(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Return one missingness row per column."""
    rows = len(frame)

    return pd.DataFrame(
        {
            "stage": stage,
            "column": frame.columns,
            "dtype": [str(frame[column].dtype) for column in frame.columns],
            "rows": rows,
            "non_null_count": [int(frame[column].notna().sum()) for column in frame.columns],
            "missing_count": [int(frame[column].isna().sum()) for column in frame.columns],
            "missing_pct": [
                round(frame[column].isna().mean() * 100, 6)
                for column in frame.columns
            ],
            "unique_count": [
                int(frame[column].nunique(dropna=True))
                for column in frame.columns
            ],
        }
    )


def apply_column_removal_rules(
    frame: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply only explicit, documented column-removal rules."""
    output = frame.copy()
    actions: list[dict[str, Any]] = []

    for column, specification in rules.items():
        rule = specification.get("rule")
        reason = specification.get("reason", "")
        exists = column in output.columns

        if not exists:
            actions.append(
                {
                    "column": column,
                    "action": "not_found",
                    "rule": rule,
                    "missing_pct_before": pd.NA,
                    "reason": reason,
                }
            )
            continue

        missing_pct = float(output[column].isna().mean() * 100)
        should_drop = False

        if rule == "drop_if_present":
            should_drop = True
        elif rule == "drop_if_missing_pct_at_least":
            threshold = float(specification["threshold_pct"])
            should_drop = missing_pct >= threshold
        else:
            raise ValueError(f"Unsupported removal rule for '{column}': {rule}")

        if should_drop:
            output = output.drop(columns=[column])
            action = "dropped"
        else:
            action = "retained_rule_not_met"

        actions.append(
            {
                "column": column,
                "action": action,
                "rule": rule,
                "missing_pct_before": round(missing_pct, 6),
                "reason": reason,
            }
        )

    return output, pd.DataFrame(actions)


def apply_semantic_fills(
    frame: pd.DataFrame,
    fills: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill only nulls with an explicit semantic meaning."""
    output = frame.copy()
    actions: list[dict[str, Any]] = []

    for column, replacement in fills.items():
        if column not in output.columns:
            actions.append(
                {
                    "column": column,
                    "replacement": replacement,
                    "values_filled": 0,
                    "status": "column_not_found",
                }
            )
            continue

        missing_before = int(output[column].isna().sum())
        output[column] = output[column].fillna(replacement)

        actions.append(
            {
                "column": column,
                "replacement": replacement,
                "values_filled": missing_before,
                "status": "filled" if missing_before else "no_missing_values",
            }
        )

    return output, pd.DataFrame(actions)


def standardize_types(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply compact and semantically appropriate data types."""
    output = frame.copy()

    timestamp_columns = [
        "timestamp",
        "timestamp_utc",
        "timestamp_toronto",
        "date",
    ]

    for column in timestamp_columns:
        if column in output.columns:
            if column == "timestamp_utc":
                output[column] = pd.to_datetime(
                    output[column], errors="coerce", utc=True
                )
            else:
                output[column] = pd.to_datetime(output[column], errors="coerce")

    for column in config["cleaning"].get("binary_columns", []):
        if column in output.columns:
            numeric = pd.to_numeric(output[column], errors="coerce")
            output[column] = numeric.astype("Int8")

    compact_integer_types = {
        "year": "Int16",
        "quarter": "Int8",
        "month": "Int8",
        "week_of_year": "Int8",
        "day_of_year": "Int16",
        "day_of_month": "Int8",
        "hour": "Int8",
        "weekday": "Int8",
        "utc_offset_hours": "Int8",
        "date_index": "Int32",
        "assigned_climate_id": "Int64",
        "reported_premise_count": "Int64",
        "source_segment_count": "Int8",
        "customer_type_count": "Int8",
        "price_plan_count": "Int8",
        "days_to_holiday": "Int16",
        "days_after_holiday": "Int16",
    }

    for column, dtype in compact_integer_types.items():
        if column in output.columns:
            output[column] = pd.to_numeric(
                output[column], errors="coerce"
            ).astype(dtype)

    for column in config["cleaning"].get("categorical_columns", []):
        if column in output.columns:
            output[column] = output[column].astype("category")

    return output


def build_consistency_checks(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate hard and informational consistency rules."""
    checks: list[dict[str, Any]] = []
    cleaning = config["cleaning"]
    key = cleaning["required_key"]

    def add_check(
        check_name: str,
        violations: int,
        severity: str,
        description: str,
    ) -> None:
        checks.append(
            {
                "check": check_name,
                "violations": int(violations),
                "severity": severity,
                "status": "PASS" if violations == 0 else "FAIL",
                "description": description,
            }
        )

    missing_key = int(frame[key].isna().any(axis=1).sum())
    add_check(
        "required_key_not_null",
        missing_key,
        "error",
        "FSA and timestamp must be present for every row.",
    )

    duplicate_key = int(frame.duplicated(subset=key, keep=False).sum())
    add_check(
        "unique_fsa_timestamp",
        duplicate_key,
        "error",
        "The cleaned dataset must contain one row per FSA and timestamp.",
    )

    for column in cleaning["required_non_null"]:
        violations = int(frame[column].isna().sum()) if column in frame.columns else len(frame)
        add_check(
            f"required_non_null::{column}",
            violations,
            "error",
            f"Required field '{column}' must not be missing.",
        )

    if "total_consumption_kwh" in frame.columns:
        add_check(
            "non_negative_consumption",
            int((frame["total_consumption_kwh"] < 0).sum()),
            "error",
            "Electricity consumption must not be negative.",
        )

    if "reported_premise_count" in frame.columns:
        add_check(
            "non_negative_premise_count",
            int((frame["reported_premise_count"] < 0).sum()),
            "error",
            "Reported premise count must not be negative.",
        )

    if "Rel Hum (%)" in frame.columns:
        humidity = frame["Rel Hum (%)"]
        add_check(
            "relative_humidity_range",
            int(((humidity < 0) | (humidity > 100)).fillna(False).sum()),
            "error",
            "Relative humidity must be between 0 and 100 percent.",
        )

    non_negative_weather = [
        "Precip. Amount (mm)",
        "Wind Spd (km/h)",
        "Visibility (km)",
    ]
    for column in non_negative_weather:
        if column in frame.columns:
            add_check(
                f"non_negative::{column}",
                int((frame[column] < 0).fillna(False).sum()),
                "error",
                f"'{column}' must not contain negative values.",
            )

    if "Wind Dir (10s deg)" in frame.columns:
        direction = frame["Wind Dir (10s deg)"]
        add_check(
            "wind_direction_range",
            int(((direction < 0) | (direction > 36)).fillna(False).sum()),
            "error",
            "Wind direction in tens of degrees must be between 0 and 36.",
        )

    if {"timestamp", "year", "month", "day_of_month", "hour"}.issubset(frame.columns):
        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
        add_check(
            "timestamp_year_consistency",
            int((timestamp.dt.year != frame["year"]).fillna(False).sum()),
            "error",
            "Calendar year must agree with timestamp.",
        )
        add_check(
            "timestamp_month_consistency",
            int((timestamp.dt.month != frame["month"]).fillna(False).sum()),
            "error",
            "Calendar month must agree with timestamp.",
        )
        add_check(
            "timestamp_day_consistency",
            int((timestamp.dt.day != frame["day_of_month"]).fillna(False).sum()),
            "error",
            "Calendar day must agree with timestamp.",
        )
        add_check(
            "timestamp_hour_consistency",
            int((timestamp.dt.hour != frame["hour"]).fillna(False).sum()),
            "error",
            "Calendar hour must agree with timestamp.",
        )

    for column in cleaning.get("binary_columns", []):
        if column in frame.columns:
            invalid = ~frame[column].isin([0, 1]) & frame[column].notna()
            add_check(
                f"binary_domain::{column}",
                int(invalid.sum()),
                "error",
                f"Binary field '{column}' must contain only 0 or 1.",
            )

    if {"is_dst_transition_day", "fsa", "timestamp"}.issubset(frame.columns):
        dst = frame.loc[frame["is_dst_transition_day"] == 1].copy()
        if not dst.empty:
            dst["date_check"] = pd.to_datetime(dst["timestamp"]).dt.date
            counts = dst.groupby(["fsa", "date_check"])["timestamp"].nunique()
            violations = int((counts != 24).sum())
        else:
            violations = 0

        add_check(
            "dst_transition_days_have_24_source_hours",
            violations,
            "error",
            "Every FSA must preserve 24 source-aligned hours on DST transition dates.",
        )

    return pd.DataFrame(checks)


def build_outlier_report(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Report IQR outliers without modifying the values."""
    rows: list[dict[str, Any]] = []

    for column in columns:
        if column not in frame.columns:
            continue

        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)

        rows.append(
            {
                "column": column,
                "non_null_count": len(series),
                "minimum": series.min(),
                "p01": series.quantile(0.01),
                "q1": q1,
                "median": series.median(),
                "q3": q3,
                "p99": series.quantile(0.99),
                "maximum": series.max(),
                "iqr_lower_bound": lower,
                "iqr_upper_bound": upper,
                "iqr_outlier_count": int(mask.sum()),
                "iqr_outlier_pct": round(mask.mean() * 100, 6),
                "action": "reported_only_not_removed",
            }
        )

    return pd.DataFrame(rows)


def build_fsa_consumption_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Profile the cleaned target by FSA without changing it."""
    return (
        frame.groupby("fsa", observed=True)
        .agg(
            rows=("timestamp", "size"),
            unique_hours=("timestamp", "nunique"),
            minimum_timestamp=("timestamp", "min"),
            maximum_timestamp=("timestamp", "max"),
            minimum_consumption_kwh=("total_consumption_kwh", "min"),
            median_consumption_kwh=("total_consumption_kwh", "median"),
            mean_consumption_kwh=("total_consumption_kwh", "mean"),
            maximum_consumption_kwh=("total_consumption_kwh", "max"),
            missing_consumption=("total_consumption_kwh", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )


def write_cleaning_report(
    output_path: Path,
    input_path: Path,
    clean_path: Path,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    column_actions: pd.DataFrame,
    semantic_actions: pd.DataFrame,
    missingness_after: pd.DataFrame,
    consistency: pd.DataFrame,
    outliers: pd.DataFrame,
) -> None:
    high_missing = missingness_after.loc[
        missingness_after["missing_pct"] >= 10
    ].sort_values("missing_pct", ascending=False)

    content = f"""# Data Cleaning Report

## Scope

Input: `{input_path}`  
Output: `{clean_path}`

The integrated Master Dataset was preserved. A separate cleaned dataset was
created for EDA and subsequent processing.

## Dataset dimensions

- Before cleaning: **{before_shape[0]:,} rows × {before_shape[1]} columns**
- After cleaning: **{after_shape[0]:,} rows × {after_shape[1]} columns**

No row was removed as part of routine missing-value or outlier treatment.

## Column actions

{as_markdown(column_actions)}

## Semantic missing-value actions

{as_markdown(semantic_actions)}

## Remaining columns with at least 10% missing values

{as_markdown(high_missing)}

These weather fields were retained because their missingness may be
condition-dependent or station-dependent. They were not globally imputed.

## Consistency checks

{as_markdown(consistency)}

## Outlier review

{as_markdown(outliers)}

Outliers were not removed. High electricity demand is analytically meaningful
for both forecasting and Peak-Risk classification. The IQR results are
diagnostic flags, not automatic deletion rules.

## Missing-value policy

- Required keys and the demand target must not be missing.
- `holiday_name` is filled with `No holiday`, because its null value has a clear
  semantic meaning.
- Numeric weather variables remain missing at this phase.
- Any model requiring complete numeric inputs must fit its imputation logic
  using only the training portion of each rolling/expanding validation window.

## Feature-engineering boundary

No lag, rolling, interaction, scaling, encoded, Peak-Risk, or forecast-horizon
variables were created during this phase.
"""
    output_path.write_text(content, encoding="utf-8")


def clean_master_dataset(
    config_path: str | Path = "configs/data_cleaning.yaml",
) -> pd.DataFrame:
    """Run the complete Phase 4 cleaning pipeline."""
    config, project_root = load_config(config_path)

    input_path = resolve_project_path(
        project_root, config["paths"]["input_master_dataset"]
    )
    output_path = resolve_project_path(
        project_root, config["paths"]["output_clean_dataset"]
    )
    reports_dir = resolve_project_path(
        project_root, config["paths"]["reports_dir"]
    )
    docs_dir = resolve_project_path(
        project_root, config["paths"]["docs_dir"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    original = load_master_dataset(input_path)
    before_shape = original.shape
    missing_before = profile_missingness(original, "before_cleaning")

    cleaned, column_actions = apply_column_removal_rules(
        original,
        config["cleaning"].get("columns_to_drop", {}),
    )

    cleaned, semantic_actions = apply_semantic_fills(
        cleaned,
        config["cleaning"].get("semantic_fill", {}),
    )

    cleaned = standardize_types(cleaned, config)
    cleaned = cleaned.sort_values(["fsa", "timestamp"], kind="stable").reset_index(drop=True)

    consistency = build_consistency_checks(cleaned, config)
    error_failures = consistency.loc[
        (consistency["severity"] == "error")
        & (consistency["violations"] > 0)
    ]

    missing_after = profile_missingness(cleaned, "after_cleaning")
    outliers = build_outlier_report(
        cleaned,
        config["cleaning"].get("outlier_review_columns", []),
    )
    fsa_summary = build_fsa_consumption_summary(cleaned)

    pd.concat([missing_before, missing_after], ignore_index=True).to_csv(
        reports_dir / "missingness_before_after.csv",
        index=False,
        encoding="utf-8-sig",
    )
    column_actions.to_csv(
        reports_dir / "column_actions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    semantic_actions.to_csv(
        reports_dir / "semantic_missing_value_actions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    consistency.to_csv(
        reports_dir / "consistency_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    outliers.to_csv(
        reports_dir / "outlier_review.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fsa_summary.to_csv(
        reports_dir / "consumption_summary_by_fsa.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not error_failures.empty:
        raise ValueError(
            "The cleaned dataset failed one or more hard consistency checks. "
            f"Review: {reports_dir / 'consistency_checks.csv'}"
        )

    cleaned.to_parquet(output_path, index=False)

    write_cleaning_report(
        docs_dir / "Data_Cleaning_Report.md",
        input_path,
        output_path,
        before_shape,
        cleaned.shape,
        column_actions,
        semantic_actions,
        missing_after,
        consistency,
        outliers,
    )

    print(f"Created cleaned dataset: {output_path.resolve()}")
    print(f"Rows: {len(cleaned):,}")
    print(f"Columns: {cleaned.shape[1]}")
    print(f"Reports: {reports_dir.resolve()}")
    print(f"Documentation: {(docs_dir / 'Data_Cleaning_Report.md').resolve()}")

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the Capstone Master Dataset.")
    parser.add_argument(
        "--config",
        default="configs/data_cleaning.yaml",
        help="Path to the Phase 4 YAML configuration.",
    )
    args = parser.parse_args()
    clean_master_dataset(args.config)


if __name__ == "__main__":
    main()
