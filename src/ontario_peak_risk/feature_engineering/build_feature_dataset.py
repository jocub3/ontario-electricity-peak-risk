"""Execute the complete Phase 6 Feature Engineering pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .base_features import build_base_dataset
from .common import (
    dataframe_to_markdown,
    ensure_directories,
    load_clean_dataset,
    load_feature_config,
    resolve_project_path,
    save_table,
)
from .feature_selection import (
    high_correlation_pairs,
    preliminary_feature_selection,
)
from .interaction_features import add_interaction_features
from .lag_features import add_target_lags
from .multicollinearity import (
    correlation_matrix,
    vif_analysis,
)
from .peak_features import peak_feature_design
from .rolling_features import add_target_rolling_features
from .temporal_features import add_temporal_features
from .validation import validate_feature_dataset
from .weather_features import add_weather_features


def feature_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create a statistical feature summary."""
    rows = []

    for column in frame.columns:
        series = frame[column]
        row = {
            "feature": column,
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(series.isna().mean() * 100, 6),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            row.update(
                {
                    "mean": series.mean(),
                    "median": series.median(),
                    "minimum": series.min(),
                    "maximum": series.max(),
                }
            )
        else:
            row.update(
                {
                    "mean": pd.NA,
                    "median": pd.NA,
                    "minimum": pd.NA,
                    "maximum": pd.NA,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def write_design_document(
    docs_dir: Path,
    config: dict,
    peak_design: pd.DataFrame,
) -> None:
    """Document the Feature Engineering design before transformation."""
    target = config["feature_engineering"]["target_column"]
    lags = config["feature_engineering"]["lags"]["target_lags_hours"]
    windows = config["feature_engineering"]["rolling"]["target_windows_hours"]

    content = f"""# Feature Engineering Design

## Objective

The Feature Engineering Phase transforms the cleaned analytical dataset into reproducible predictor
variables for the Forecasting and Peak-Risk models while preserving temporal
ordering and preventing data leakage.

## Shared feature families

Both modeling tasks may use:

- existing calendar and FSA variables;
- cyclical temporal encodings;
- current/future weather inputs available at prediction time;
- historical electricity-demand lags;
- leakage-safe rolling demand summaries;
- weather change and rolling summaries;
- a small set of EDA-justified interaction features.

## Forecasting target

`{target}` remains the observed target. No future forecast-horizon targets are
created during this phase.

## Historical target features

Configured lags: `{lags}`

Configured rolling windows: `{windows}`

All target rolling features are calculated after a one-hour shift, so the
current target value is excluded.

## Peak-Risk design

The final Peak-Risk label is intentionally NOT generated in the static Feature
Engineering dataset. Seasonal/FSA thresholds must be estimated using training
data only inside each temporal validation window.

{dataframe_to_markdown(peak_design)}

## Forecast-Horizon Availability

Historical demand features are valid only when the information required to calculate them 
is available at the time each forecast horizon is generated.

For a 24-hour forecast, the availability of short lags depends on the forecasting strategy. 
For example, `lag_1h` is directly available for the first forecasted hour, but future horizons 
may require either recursively predicted demand values or a direct/multi-output design that 
uses only information available at forecast origin.

Therefore, historical lag and rolling features created during this phase are candidate predictors. 
Their actual availability and construction must be validated separately for each forecasting strategy 
and forecast horizon during the modeling phase.

## Leakage policy

Features are classified into three calculation scopes:

- `static_dataset`: deterministic from timestamp, geography, or existing data;
- `current_exogenous`: may use weather information assumed available from the
  operational weather forecast;
- `historical_only`: must only use observations earlier than the prediction time.

No future observed electricity-demand value is used as a predictor.
"""
    (docs_dir / "06_00_Feature_Engineering_Design.md").write_text(
        content,
        encoding="utf-8",
    )


def write_summary_document(
    docs_dir: Path,
    feature_dictionary: pd.DataFrame,
    selection: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Write the Phase 6 completion summary."""
    # Count each final feature only once using its most recent classification.
    # Engineered classifications must override the original baseline classification.
    family_counts = (
        feature_dictionary
        .drop_duplicates(
            subset=["feature_name"],
            keep="last",
        )
        .groupby(
            "feature_family",
            observed=True,
        )
        .size()
        .rename("feature_count")
        .reset_index()
        .sort_values("feature_family")
        .reset_index(drop=True)
    )

    priority_counts = (
        selection.groupby("preliminary_priority", observed=True)
        .size()
        .rename("feature_count")
        .reset_index()
    )

    content = f"""# Feature Engineering Summary
    
    ## Feature families

    {dataframe_to_markdown(family_counts)}

    ## Preliminary feature priorities

    {dataframe_to_markdown(priority_counts)}

    ## Validation

    {dataframe_to_markdown(validation)}

    ## Modeling handoff

    `feature_dataset.parquet` is the static feature table produced by the Feature Engineering Phase.

    Modeling code must still:

    1. perform chronological train/validation/test splits;
    2. fit missing-value treatment using training data only;
    3. create the final Peak-Risk thresholds and labels inside each training window;
    4. ensure that future weather inputs correspond to forecast-available weather,
    not future observed weather;
    5. select final predictors separately for each candidate model.

    ## Forecast-Horizon Constraint

    Historical demand features must be reviewed for each forecast horizon before model training. 
    Their presence in `feature_dataset.parquet` does not imply that they are directly available for all 24 future hours.
    """
    (docs_dir / "06_06_Feature_Engineering_Summary.md").write_text(
        content,
        encoding="utf-8",
    )


def build_feature_dataset(
    config_path: str | Path = "configs/feature_engineering.yaml",
) -> pd.DataFrame:
    """Run all Phase 6 transformations and reports."""
    config, project_root = load_feature_config(config_path)
    reports_dir, docs_dir = ensure_directories(config, project_root)

    clean = load_clean_dataset(config, project_root)
    base, base_log = build_base_dataset(clean, config)

    base_path = resolve_project_path(
        project_root,
        config["paths"]["base_dataset"],
    )
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(base_path, index=False)

    temporal, temporal_log = add_temporal_features(base, config)
    lagged, lag_log = add_target_lags(temporal, config)
    rolled, rolling_log = add_target_rolling_features(lagged, config)
    weathered, weather_log = add_weather_features(rolled, config)
    engineered, interaction_log = add_interaction_features(weathered, config)

    feature_dictionary = pd.concat(
        [
            base_log,
            temporal_log,
            lag_log,
            rolling_log,
            weather_log,
            interaction_log,
        ],
        ignore_index=True,
    )

    peak_design = peak_feature_design()
    write_design_document(docs_dir, config, peak_design)

    selection = preliminary_feature_selection(
        engineered,
        feature_dictionary,
        config,
    )
    high_corr = high_correlation_pairs(engineered, config)

    pearson = correlation_matrix(engineered, "pearson")
    spearman = correlation_matrix(engineered, "spearman")

    try:
        vif = vif_analysis(engineered, config)
    except Exception as exc:
        vif = pd.DataFrame(
            [
                {
                    "feature": "VIF_NOT_AVAILABLE",
                    "vif": pd.NA,
                    "note": str(exc),
                }
            ]
        )

    validation = validate_feature_dataset(
        base,
        engineered,
        config,
    )

    failures = validation.loc[validation["status"] == "FAIL"]
    if not failures.empty:
        save_table(
            validation,
            reports_dir / "06_05_feature_validation.csv",
        )
        raise ValueError(
            "Feature Engineering validation failed. "
            "Review reports/feature_engineering/06_05_feature_validation.csv"
        )

    final_path = resolve_project_path(
        project_root,
        config["paths"]["final_dataset"],
    )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_parquet(final_path, index=False)

    summary = feature_summary(engineered)
    missingness = summary[
        [
            "feature",
            "dtype",
            "non_null_count",
            "missing_count",
            "missing_pct",
        ]
    ].copy()

    save_table(
        feature_dictionary,
        reports_dir / "06_01_feature_dictionary.csv",
    )
    save_table(
        summary,
        reports_dir / "06_02_feature_summary.csv",
    )
    save_table(
        missingness,
        reports_dir / "06_03_feature_missingness.csv",
    )
    save_table(
        pearson.reset_index(),
        reports_dir / "06_04_feature_correlation_pearson.csv",
    )
    save_table(
        spearman.reset_index(),
        reports_dir / "06_04_feature_correlation_spearman.csv",
    )
    save_table(
        high_corr,
        reports_dir / "06_04_high_correlation_pairs.csv",
    )
    save_table(
        vif,
        reports_dir / "06_04_vif_analysis.csv",
    )
    save_table(
        validation,
        reports_dir / "06_05_feature_validation.csv",
    )
    save_table(
        selection,
        reports_dir / "06_07_preliminary_feature_selection.csv",
    )
    save_table(
        peak_design,
        reports_dir / "06_07_peak_feature_design.csv",
    )

    # Temporary diagnostic: verify feature-family counts before writing the summary
    print(
        "\nFeature family counts before generating the summary:"
    )

    print(
        feature_dictionary
        .drop_duplicates(subset=["feature_name"], keep="last")
        ["feature_family"]
        .value_counts()
    )

    write_summary_document(
        docs_dir,
        feature_dictionary,
        selection,
        validation,
    )

    print(f"Base dataset: {base_path.resolve()}")
    print(f"Final feature dataset: {final_path.resolve()}")
    print(f"Rows: {len(engineered):,}")
    print(f"Columns: {engineered.shape[1]}")
    print(f"Reports: {reports_dir.resolve()}")
    print(f"Documentation: {docs_dir.resolve()}")

    return engineered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Capstone Feature Dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/feature_engineering.yaml",
    )
    args = parser.parse_args()
    build_feature_dataset(args.config)


if __name__ == "__main__":
    main()
