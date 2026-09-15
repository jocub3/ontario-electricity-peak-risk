
"""Generate the project Data Dictionary from the complete datasets.

Outputs:
- reports/data_dictionary.csv
- docs/Data_Dictionary.md

The script combines:
1. curated, source-grounded definitions;
2. technical metadata observed in the actual project files
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.ontario_peak_risk.utils.io import (
    ensure_output_directories,
    load_project_datasets,
)


IESO_REFERENCE = (
    "Independent Electricity System Operator. Hourly Electricity "
    "Consumption Data by Forward Sortation Area report description."
)
ECCC_REFERENCE = (
    "Environment and Climate Change Canada. Historical Climate Data "
    "and Technical Documentation."
)
PROJECT_REFERENCE = (
    "Project-generated variable. Definition based on the calendar or "
    "preprocessing script stored in this repository."
)


def _field(
    description: str,
    *,
    unit: str = "",
    role: str = "",
    source_reference: str = "",
    notes: str = "",
) -> dict[str, str]:
    return {
        "description": description,
        "unit": unit,
        "analytical_role": role,
        "source_reference": source_reference,
        "notes": notes,
    }


# Definitions are intentionally conservative.
# Flag-code meanings are not inferred from the sample values.
DICTIONARY: dict[str, dict[str, dict[str, str]]] = {
    "consumption": {
        "FSA": _field(
            "Forward Sortation Area: the first three characters of a Canadian postal code.",
            role="Geographic identifier / grouping variable",
            source_reference=IESO_REFERENCE,
        ),
        "DATE": _field(
            "Source date associated with the IESO hourly record.",
            role="Temporal identifier",
            source_reference=IESO_REFERENCE,
            notes="Interpret together with HOUR and the documented hour-ending convention.",
        ),
        "HOUR": _field(
            "IESO hour-ending number, represented from 1 through 24.",
            role="Temporal identifier",
            source_reference=IESO_REFERENCE,
        ),
        "CUSTOMER_TYPE": _field(
            "Customer category used by IESO for the aggregated record, including Residential and Small General Service below 50 kW where present.",
            role="Segmentation dimension",
            source_reference=IESO_REFERENCE,
        ),
        "PRICE_PLAN": _field(
            "Electricity price-plan category associated with the aggregated record, such as TOU, Tiered, or Retailer where present.",
            role="Segmentation dimension",
            source_reference=IESO_REFERENCE,
        ),
        "TOTAL_CONSUMPTION": _field(
            "Aggregated electricity consumption for the premises represented by the record.",
            unit="kWh",
            role="Primary demand measure / future forecasting target after aggregation",
            source_reference=IESO_REFERENCE,
        ),
        "PREMISE_COUNT": _field(
            "Number of premises included in the aggregated line record.",
            unit="premises",
            role="Coverage and normalization measure",
            source_reference=IESO_REFERENCE,
            notes="It is not a population count and may reflect source publication and quality rules.",
        ),
        "SOURCE_PERIOD": _field(
            "Monthly source period extracted during preprocessing.",
            role="Traceability field",
            source_reference=PROJECT_REFERENCE,
            notes="Expected format: YYYYMM.",
        ),
        "INTERVAL_START_TIMESTAMP": _field(
            "Start of the hourly consumption interval derived from DATE and the IESO hour-ending value.",
            role="Primary temporal integration key",
            source_reference=PROJECT_REFERENCE,
        ),
        "INTERVAL_END_TIMESTAMP": _field(
            "End of the hourly consumption interval derived from DATE and the IESO hour-ending value.",
            role="Temporal audit field",
            source_reference=PROJECT_REFERENCE,
        ),
        "TIMESTAMP": _field(
            "Project-selected timestamp used for integration; currently aligned with the interval start.",
            role="Primary temporal integration key",
            source_reference=PROJECT_REFERENCE,
        ),
    },
    "weather": {
        "Longitude (x)": _field(
            "Longitude of the climate station.",
            unit="decimal degrees",
            role="Station metadata",
            source_reference=ECCC_REFERENCE,
        ),
        "Latitude (y)": _field(
            "Latitude of the climate station.",
            unit="decimal degrees",
            role="Station metadata",
            source_reference=ECCC_REFERENCE,
        ),
        "Station Name": _field(
            "Name of the ECCC/NAVCAN climate station.",
            role="Station identifier",
            source_reference=ECCC_REFERENCE,
        ),
        "Climate ID": _field(
            "Climate station identifier used by ECCC.",
            role="Station identifier",
            source_reference=ECCC_REFERENCE,
        ),
        "Date/Time (LST)": _field(
            "Date and time of the hourly observation in Local Standard Time as labelled by ECCC.",
            role="Primary temporal integration key",
            source_reference=ECCC_REFERENCE,
            notes="Do not silently reinterpret as daylight-adjusted Toronto civil time.",
        ),
        "Year": _field("Observation year.", role="Temporal feature", source_reference=ECCC_REFERENCE),
        "Month": _field("Observation month.", role="Temporal feature", source_reference=ECCC_REFERENCE),
        "Day": _field("Observation day of month.", role="Temporal feature", source_reference=ECCC_REFERENCE),
        "Time (LST)": _field(
            "Time component of the Local Standard Time observation.",
            role="Temporal feature",
            source_reference=ECCC_REFERENCE,
        ),
        "Flag": _field(
            "General source quality/status flag supplied by ECCC.",
            role="Data-quality metadata",
            source_reference=ECCC_REFERENCE,
            notes="Exact code meaning must be verified in the ECCC documentation before interpretation.",
        ),
        "Temp (°C)": _field(
            "Hourly air temperature.",
            unit="°C",
            role="Core weather predictor",
            source_reference=ECCC_REFERENCE,
        ),
        "Temp Flag": _field(
            "ECCC flag associated with air temperature.",
            role="Data-quality metadata",
            source_reference=ECCC_REFERENCE,
            #notes="Exact flag-code meanings require manual verification.",
            notes="",
        ),
        "Dew Point Temp (°C)": _field(
            "Hourly dew-point temperature.",
            unit="°C",
            role="Weather predictor",
            source_reference=ECCC_REFERENCE,
        ),
        "Dew Point Temp Flag": _field(
            "ECCC flag associated with dew-point temperature.",
            role="Data-quality metadata",
            source_reference=ECCC_REFERENCE,
            #notes="Exact flag-code meanings require manual verification.",
            notes=""
        ),
        "Rel Hum (%)": _field(
            "Hourly relative humidity.",
            unit="%",
            role="Core weather predictor",
            source_reference=ECCC_REFERENCE,
        ),
        "Rel Hum Flag": _field(
            "ECCC flag associated with relative humidity.",
            role="Data-quality metadata",
            source_reference=ECCC_REFERENCE,
            #notes="Exact flag-code meanings require manual verification.",
            notes=""
        ),
        "Precip. Amount (mm)": _field(
            "Hourly precipitation amount.",
            unit="mm",
            role="Weather predictor",
            source_reference=ECCC_REFERENCE,
        ),
        "Precip. Amount Flag": _field(
            "ECCC flag associated with precipitation amount.",
            role="Data-quality metadata",
            source_reference=ECCC_REFERENCE,
            #notes="Exact flag-code meanings require manual verification.",
            notes=""
        ),
        "Wind Dir (10s deg)": _field(
            "Wind direction represented in tens of degrees.",
            unit="10 degrees",
            role="Weather predictor",
            source_reference=ECCC_REFERENCE,
        ),
        "Wind Dir Flag": _field(
            "ECCC flag associated with wind direction.",
            role="Data-quality metadata",
            source_reference=ECCC_REFERENCE,
            #notes="Exact flag-code meanings require manual verification.",
            notes=""
        ),
        "Wind Spd (km/h)": _field(
            "Hourly wind speed.",
            unit="km/h",
            role="Weather predictor",
            source_reference=ECCC_REFERENCE,
        ),
        "Wind Spd Flag": _field(
            "ECCC flag associated with wind speed.",
            role="Data-quality metadata",
            source_reference=ECCC_REFERENCE,
            #notes="Exact flag-code meanings require manual verification.",
            notes=""
        ),
        "Visibility (km)": _field(
            "Horizontal visibility.",
            unit="km",
            role="Optional weather predictor",
            source_reference=ECCC_REFERENCE,
        ),
        "Visibility Flag": _field(
            "ECCC flag associated with visibility.",
            role="Data-quality metadata",
            source_reference=ECCC_REFERENCE,
            #notes="Exact flag-code meanings require manual verification.",
            notes=""
        ),
        "Stn Press (kPa)": _field(
            "Atmospheric pressure measured at the station.",
            unit="kPa",
            role="Weather predictor",
            source_reference=ECCC_REFERENCE,
        ),
        "Stn Press Flag": _field(
            "ECCC flag associated with station pressure.",
            role="Data-quality metadata",
            source_reference=ECCC_REFERENCE,
            #notes="Exact flag-code meanings require manual verification.",
            notes=""
        ),
        "Hmdx": _field(
            "Humidex value when reported by the source.",
            role="Derived heat-stress predictor",
            source_reference=ECCC_REFERENCE,
        ),
        "Hmdx Flag": _field(
            "ECCC flag associated with humidex.",
            role="Data-quality metadata",
            source_reference=ECCC_REFERENCE,
            #notes="Exact flag-code meanings require manual verification.",
            notes=""
        ),
        "Wind Chill": _field(
            "Wind-chill value when reported by the source.",
            role="Derived cold-stress predictor",
            source_reference=ECCC_REFERENCE,
        ),
        "Wind Chill Flag": _field(
            "ECCC flag associated with wind chill.",
            role="Data-quality metadata",
            source_reference=ECCC_REFERENCE,
            #notes="Exact flag-code meanings require manual verification.",
            notes=""
        ),
        "Weather": _field(
            "Text description of observed weather conditions when reported.",
            role="Optional categorical weather predictor",
            source_reference=ECCC_REFERENCE,
        ),
        "SOURCE_FILE": _field(
            "Original monthly weather filename.",
            role="Traceability field",
            source_reference=PROJECT_REFERENCE,
        ),
        "SOURCE_PERIOD": _field(
            "Monthly source period extracted during preprocessing.",
            role="Traceability field",
            source_reference=PROJECT_REFERENCE,
            notes="Expected format: YYYYMM.",
        ),
    },
    "calendar": {
        "timestamp_utc": _field(
            "UTC representation derived from the fixed source-aligned local-standard-time key.",
            role="Universal temporal audit key",
            source_reference=PROJECT_REFERENCE,
        ),
        "timestamp_local": _field(
            "Source-aligned hourly key with exactly 24 records per date; used to join consumption, weather, and calendar.",
            role="Primary temporal integration key",
            source_reference=PROJECT_REFERENCE,
        ),
        "timestamp_toronto": _field(
            "Toronto civil-time representation of the UTC instant, including daylight-saving changes.",
            role="Civil-time interpretation field",
            source_reference=PROJECT_REFERENCE,
        ),
        "date": _field("Calendar date.", role="Temporal feature", source_reference=PROJECT_REFERENCE),
        "year": _field("Calendar year.", role="Temporal feature", source_reference=PROJECT_REFERENCE),
        "quarter": _field("Calendar quarter from 1 to 4.", role="Temporal feature", source_reference=PROJECT_REFERENCE),
        "month": _field("Calendar month from 1 to 12.", role="Temporal feature", source_reference=PROJECT_REFERENCE),
        "month_name": _field("English month name.", role="Descriptive temporal feature", source_reference=PROJECT_REFERENCE),
        "week_of_year": _field("ISO week number.", role="Temporal feature", source_reference=PROJECT_REFERENCE),
        "day_of_year": _field("Sequential day number within the year.", role="Temporal feature", source_reference=PROJECT_REFERENCE),
        "day_of_month": _field("Day number within the month.", role="Temporal feature", source_reference=PROJECT_REFERENCE),
        "hour": _field("Hour of day from 0 to 23.", role="Core temporal predictor", source_reference=PROJECT_REFERENCE),
        "hour_group": _field("Grouped period of day: Night, Morning, Afternoon, or Evening.", role="Categorical temporal predictor", source_reference=PROJECT_REFERENCE),
        "weekday": _field("Weekday number where Monday is 0 and Sunday is 6.", role="Core temporal predictor", source_reference=PROJECT_REFERENCE),
        "weekday_name": _field("English weekday name.", role="Descriptive temporal feature", source_reference=PROJECT_REFERENCE),
        "is_weekend": _field("Indicator equal to 1 for Saturday or Sunday.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_workday": _field("Indicator equal to 1 when the date is neither a weekend nor an Ontario public holiday.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_monday": _field("Indicator equal to 1 on Monday.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_friday": _field("Indicator equal to 1 on Friday.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "season": _field("Meteorological season derived from month.", role="Categorical temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_business_hour": _field("Project-defined weekday business-hour indicator for hours 08 through 17.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_peak_hour_window": _field("Project-defined descriptive indicator for hours 07–10 or 17–21.", role="Exploratory temporal predictor", source_reference=PROJECT_REFERENCE, notes="This is not the Peak-Risk target."),
        "is_month_start": _field("Indicator for the first calendar day of a month.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_month_end": _field("Indicator for the last calendar day of a month.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_year_start": _field("Indicator for January 1.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_year_end": _field("Indicator for December 31.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_public_holiday": _field("Indicator for a public holiday generated for Ontario.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "holiday_name": _field("Holiday name returned by the Python holidays library for Ontario.", role="Categorical temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_day_before_holiday": _field("Indicator for the calendar day immediately before a public holiday.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_day_after_holiday": _field("Indicator for the calendar day immediately after a public holiday.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "days_to_holiday": _field("Number of calendar days to the next public holiday; 0 on a holiday.", unit="days", role="Numeric temporal predictor", source_reference=PROJECT_REFERENCE),
        "days_after_holiday": _field("Number of calendar days since the previous public holiday; 0 on a holiday.", unit="days", role="Numeric temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_long_weekend": _field("Project-derived indicator for a holiday connected to a weekend.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_daylight_saving_time": _field("Indicator derived from Toronto civil time showing whether DST is active.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_dst_transition_day": _field("Indicator for a date containing a Toronto UTC-offset transition.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_spring_forward_day": _field("Indicator for the spring DST transition date.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "is_fall_back_day": _field("Indicator for the fall DST transition date.", role="Binary temporal predictor", source_reference=PROJECT_REFERENCE),
        "utc_offset_hours": _field("Toronto civil-time offset from UTC for the represented instant.", unit="hours", role="Temporal metadata / possible predictor", source_reference=PROJECT_REFERENCE),
        "date_index": _field("Number of days elapsed from the first calendar date.", unit="days", role="Trend feature", source_reference=PROJECT_REFERENCE),
        "hour_sin": _field("Sine transformation of hour for cyclical encoding.", role="Cyclical temporal predictor", source_reference=PROJECT_REFERENCE),
        "hour_cos": _field("Cosine transformation of hour for cyclical encoding.", role="Cyclical temporal predictor", source_reference=PROJECT_REFERENCE),
        "weekday_sin": _field("Sine transformation of weekday for cyclical encoding.", role="Cyclical temporal predictor", source_reference=PROJECT_REFERENCE),
        "weekday_cos": _field("Cosine transformation of weekday for cyclical encoding.", role="Cyclical temporal predictor", source_reference=PROJECT_REFERENCE),
        "month_sin": _field("Sine transformation of month for cyclical encoding.", role="Cyclical temporal predictor", source_reference=PROJECT_REFERENCE),
        "month_cos": _field("Cosine transformation of month for cyclical encoding.", role="Cyclical temporal predictor", source_reference=PROJECT_REFERENCE),
    },
}


def _safe_examples(series: pd.Series, limit: int = 3) -> str:
    values = (
        series.dropna()
        .astype(str)
        .drop_duplicates()
        .head(limit)
        .tolist()
    )
    return " | ".join(values)


def profile_dictionary(
    dataset_name: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Combine curated descriptions with observed technical metadata."""
    rows: list[dict[str, Any]] = []
    curated = DICTIONARY.get(dataset_name, {})

    for column in frame.columns:
        if column == "_SOURCE_PATH":
            continue

        metadata = curated.get(
            column,
            _field(
                "Definition not available in the current project documentation. Manual verification required.",
                source_reference="Manual verification required.",
            ),
        )

        series = frame[column]
        non_null = int(series.notna().sum())
        row_count = int(len(series))

        rows.append(
            {
                "dataset": dataset_name,
                "field_name": column,
                "description": metadata["description"],
                "source_dtype_observed": str(series.dtype),
                "unit": metadata["unit"],
                "analytical_role": metadata["analytical_role"],
                "nullable_observed": bool(series.isna().any()),
                "non_null_count": non_null,
                "missing_count": row_count - non_null,
                "missing_pct": round((row_count - non_null) / row_count * 100, 4)
                if row_count
                else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "example_values": _safe_examples(series),
                "source_reference": metadata["source_reference"],
                "notes": metadata["notes"],
            }
        )

    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)


def write_markdown(
    dictionary: pd.DataFrame,
    output_path: Path,
    start_year: int,
    end_year: int,
) -> None:
    """Write a human-readable Data Dictionary."""
    sections = [
        "# Data Dictionary",
        "",
        f"**Analysis period:** {start_year}-01-01 through {end_year}-12-31.",
        "",
        "This document combines source-grounded definitions with metadata observed "
        "in the project datasets. "
        #"A field is explicitly marked for manual review when its exact meaning is not supported by the available documentation.",
        "",
        "## Source references",
        "",
        "- Independent Electricity System Operator (IESO), *Hourly Electricity "
        "Consumption Data by Forward Sortation Area – Report Description*.",
        "- Environment and Climate Change Canada (ECCC), *Historical Climate Data* "
        "and associated technical documentation.",
        "- Project preprocessing and calendar-generation scripts in this repository.",
        "",
    ]

    display_columns = [
        "field_name",
        "description",
        "source_dtype_observed",
        "unit",
        "analytical_role",
        "missing_pct",
        "unique_count",
        "example_values",
        "notes",
    ]

    for dataset_name in ("consumption", "weather", "calendar"):
        subset = dictionary.loc[dictionary["dataset"] == dataset_name, display_columns]
        sections.extend(
            [
                f"## {dataset_name.title()} dataset",
                "",
                _markdown_table(subset),
                "",
            ]
        )

    output_path.write_text("\n".join(sections), encoding="utf-8")


def build_data_dictionary(
    processed_directory: str | Path,
    project_root: str | Path,
    *,
    start_year: int = 2021,
    end_year: int = 2025,
) -> pd.DataFrame:
    datasets = load_project_datasets(
        processed_directory,
        start_year=start_year,
        end_year=end_year,
    )

    dictionary = pd.concat(
        [
            profile_dictionary("consumption", datasets.consumption),
            profile_dictionary("weather", datasets.weather),
            profile_dictionary("calendar", datasets.calendar),
        ],
        ignore_index=True,
    )

    docs_directory, reports_directory = ensure_output_directories(project_root)

    csv_path = reports_directory / "data_dictionary.csv"
    markdown_path = docs_directory / "Data_Dictionary.md"

    dictionary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_markdown(dictionary, markdown_path, start_year, end_year)

    print(f"Data dictionary CSV: {csv_path.resolve()}")
    print(f"Data dictionary Markdown: {markdown_path.resolve()}")
    print(f"Fields documented: {len(dictionary):,}")

    return dictionary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Capstone Data Dictionary.")
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
    build_data_dictionary(
        processed_directory=args.processed_dir,
        project_root=args.project_root,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()
