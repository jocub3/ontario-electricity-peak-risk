"""Build, validate, document, and save the Capstone Master Dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.ontario_peak_risk.utils.io import load_project_datasets


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Invalid YAML configuration.")
    return config


def as_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No records."
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def document_removed_columns(df: pd.DataFrame, docs_dir: Path, reports_dir: Path, start: int, end: int) -> pd.DataFrame:
    protected = {"Station Name", "Climate ID", "Date/Time (LST)", "_SOURCE_PATH"}
    columns = [c for c in df.columns if c not in protected and df[c].isna().all()]
    report = pd.DataFrame({
        "column": columns,
        "reason": [f"100% missing across the combined {start}-{end} weather dataset"] * len(columns),
        "missing_pct": [100.0] * len(columns),
    })
    report.to_csv(reports_dir / "removed_weather_columns.csv", index=False, encoding="utf-8-sig")
    text = f"""# Removed Weather Columns

**Analysis period:** {start}-01-01 through {end}-12-31.

The source weather files remain unchanged. During Master Dataset construction,
only columns that were completely empty across the combined, date-filtered
weather dataset were removed. Partially populated fields were retained.

## Removed columns

{as_markdown(report)}
"""
    (docs_dir / "Removed_Weather_Columns.md").write_text(text, encoding="utf-8")
    return report


def aggregate_consumption(df: pd.DataFrame) -> pd.DataFrame:
    required = {"FSA", "TIMESTAMP", "TOTAL_CONSUMPTION", "PREMISE_COUNT", "CUSTOMER_TYPE", "PRICE_PLAN"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing consumption columns: {sorted(missing)}")
    work = df.copy()
    work["TIMESTAMP"] = pd.to_datetime(work["TIMESTAMP"], errors="coerce")
    out = (work.groupby(["FSA", "TIMESTAMP"], as_index=False)
           .agg(total_consumption_kwh=("TOTAL_CONSUMPTION", "sum"),
                reported_premise_count=("PREMISE_COUNT", "sum"),
                source_segment_count=("TOTAL_CONSUMPTION", "size"),
                customer_type_count=("CUSTOMER_TYPE", "nunique"),
                price_plan_count=("PRICE_PLAN", "nunique"))
           .rename(columns={"FSA": "fsa", "TIMESTAMP": "timestamp"})
           .sort_values(["fsa", "timestamp"]).reset_index(drop=True))
    if out.duplicated(["fsa", "timestamp"]).any():
        raise ValueError("Aggregation did not produce a unique FSA + timestamp key.")
    return out


def prepare_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=["_SOURCE_PATH"], errors="ignore").copy()
    out["timestamp_local"] = pd.to_datetime(out["timestamp_local"], errors="coerce")
    out = out.rename(columns={"timestamp_local": "timestamp"})
    if out.duplicated(["timestamp"]).any():
        raise ValueError("Calendar timestamp is not unique.")
    return out


def prepare_weather(df: pd.DataFrame, removed: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=removed.get("column", pd.Series(dtype=str)).tolist(), errors="ignore")
    out = out.drop(columns=["_SOURCE_PATH", "Year", "Month", "Day", "Time (LST)"], errors="ignore").copy()
    out["Date/Time (LST)"] = pd.to_datetime(out["Date/Time (LST)"], errors="coerce")
    out["Climate ID"] = pd.to_numeric(out["Climate ID"], errors="coerce").astype("Int64")
    out["Station Name"] = out["Station Name"].astype("string").str.strip().str.upper()
    out = out.rename(columns={"Date/Time (LST)": "timestamp", "Climate ID": "assigned_climate_id", "Station Name": "assigned_station_name"})
    if out.duplicated(["assigned_climate_id", "timestamp"]).any():
        raise ValueError("Weather Climate ID + timestamp is not unique.")
    return out


def station_mapping(config: dict[str, Any]) -> pd.DataFrame:
    rows = [{"fsa": fsa.upper(), "assigned_climate_id": int(v["climate_id"]), "assigned_station_name": str(v["station_name"]).upper()} for fsa, v in config["fsa_station_mapping"].items()]
    return pd.DataFrame(rows)


def build_master_dataset(config_path: str | Path = "configs/data_integration.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    start, end = int(config["analysis_period"]["start_year"]), int(config["analysis_period"]["end_year"])
    #processed = Path(config["paths"]["processed_data_dir"])
    #master_path = Path(config["paths"]["master_dataset"])
    #reports_dir, docs_dir = Path(config["paths"]["reports_dir"]), Path(config["paths"]["docs_dir"])
    #master_path.parent.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True); docs_dir.mkdir(parents=True, exist_ok=True)

    # Resolve all configured paths relative to the project root.
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent

    def resolve_project_path(path_value: str) -> Path:
        path = Path(path_value)

        if path.is_absolute():
            return path

        return project_root / path


    processed = resolve_project_path(
        config["paths"]["processed_data_dir"]
    )

    master_path = resolve_project_path(
        config["paths"]["master_dataset"]
    )

    reports_dir = resolve_project_path(
        config["paths"]["reports_dir"]
    )

    docs_dir = resolve_project_path(
        config["paths"]["docs_dir"]
    )

    # Create output directories before saving files.
    master_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    docs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_project_datasets(processed, start_year=start, end_year=end)
    removed = document_removed_columns(data.weather, docs_dir, reports_dir, start, end)
    consumption = aggregate_consumption(data.consumption)
    calendar = prepare_calendar(data.calendar)
    weather = prepare_weather(data.weather, removed)
    mapping = station_mapping(config)
    mapping.to_csv(reports_dir / "fsa_weather_station_mapping.csv", index=False, encoding="utf-8-sig")

    step_rows = []
    cc = consumption.merge(calendar, on="timestamp", how="left", validate="many_to_one", indicator=True)
    step_rows.append({"step": "consumption_calendar_join", "rows_before": len(consumption), "rows_after": len(cc), "unmatched_rows": int((cc["_merge"] != "both").sum())})
    cc = cc.drop(columns="_merge").merge(mapping, on="fsa", how="left", validate="many_to_one")
    if cc["assigned_climate_id"].isna().any():
        raise ValueError("At least one FSA has no weather-station mapping.")

    master = cc.merge(weather, on=["assigned_climate_id", "assigned_station_name", "timestamp"], how="left", validate="many_to_one", indicator=True, suffixes=("", "_weather"))
    step_rows.append({"step": "weather_join", "rows_before": len(cc), "rows_after": len(master), "unmatched_rows": int((master["_merge"] != "both").sum())})
    master = master.drop(columns="_merge").sort_values(["fsa", "timestamp"]).reset_index(drop=True)

    duplicate_count = int(master.duplicated(["fsa", "timestamp"], keep=False).sum())
    coverage = master.groupby("fsa").agg(rows=("timestamp", "size"), unique_hours=("timestamp", "nunique"), minimum_timestamp=("timestamp", "min"), maximum_timestamp=("timestamp", "max")).reset_index()
    coverage["expected_hours"] = consumption.groupby("fsa")["timestamp"].nunique().reindex(coverage["fsa"]).to_numpy()
    coverage["missing_hours"] = coverage["expected_hours"] - coverage["unique_hours"]
    core_weather = [c for c in ["Temp (°C)", "Rel Hum (%)", "Stn Press (kPa)"] if c in master.columns]
    validation = pd.DataFrame([{
        "rows": len(master), "columns": master.shape[1], "unique_fsas": master["fsa"].nunique(),
        "minimum_timestamp": master["timestamp"].min(), "maximum_timestamp": master["timestamp"].max(),
        "duplicate_fsa_timestamp_rows": duplicate_count,
        "rows_without_calendar": int(master["year"].isna().sum()),
        "rows_missing_all_core_weather": int(master[core_weather].isna().all(axis=1).sum()) if core_weather else pd.NA,
        "missing_consumption": int(master["total_consumption_kwh"].isna().sum()),
        "negative_consumption": int((master["total_consumption_kwh"] < 0).sum()),
    }])
    dst = (master.loc[master.get("is_dst_transition_day", 0) == 1]
           .assign(date_check=lambda x: x["timestamp"].dt.date)
           .groupby(["fsa", "date_check"], as_index=False)
           .agg(hourly_rows=("timestamp", "size"), unique_hours=("timestamp", "nunique"), spring=("is_spring_forward_day", "max"), fall=("is_fall_back_day", "max")))

    if len(master) != len(consumption) or duplicate_count or coverage["missing_hours"].sum():
        raise ValueError("Master Dataset failed row-preservation or uniqueness validation.")

    pd.DataFrame(step_rows).to_csv(reports_dir / "data_preparation_step_summary.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(reports_dir / "master_dataset_validation_summary.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(reports_dir / "master_dataset_coverage_by_fsa.csv", index=False, encoding="utf-8-sig")
    dst.to_csv(reports_dir / "master_dataset_dst_validation.csv", index=False, encoding="utf-8-sig")
    master.to_parquet(master_path, index=False)

    report = f"""# Data Preparation and Integration Report

**Period:** {start}-01-01 through {end}-12-31.

## Decisions

- Final grain: `FSA + timestamp`.
- Consumption is aggregated by summing `TOTAL_CONSUMPTION` and `PREMISE_COUNT`.
- Calendar join key: consumption `TIMESTAMP` = calendar `timestamp_local`.
- Weather is assigned using `configs/data_integration.yaml` and joined by station + timestamp.
- Only 100%-empty weather columns are removed; source files remain unchanged.

## Join summary

{as_markdown(pd.DataFrame(step_rows))}

## Final validation

{as_markdown(validation)}

## Coverage by FSA

{as_markdown(coverage)}

## DST validation

{as_markdown(dst)}

## Output

`{master_path}` is the single integrated input for subsequent cleaning, EDA, feature engineering, and modeling.
"""
    (docs_dir / "Data_Preparation_Report.md").write_text(report, encoding="utf-8")
    print(f"Created: {master_path.resolve()} | rows={len(master):,} | columns={master.shape[1]}")
    return master


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_integration.yaml")
    args = parser.parse_args()
    build_master_dataset(args.config)


if __name__ == "__main__":
    main()
