
"""Input/output utilities for the Capstone data-understanding phase.

The loaders in this module:
- discover all matching files;
- combine files belonging to the same dataset;
- standardize important data types;
- restrict the analysis to a configurable date range;
- preserve the original columns.

The default project period is 2021-01-01 through 2025-12-31.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class AnalysisPeriod:
    """Inclusive analysis period."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp

    @classmethod
    def from_years(cls, start_year: int, end_year: int) -> "AnalysisPeriod":
        if end_year < start_year:
            raise ValueError("end_year must be greater than or equal to start_year.")

        return cls(
            start_date=pd.Timestamp(year=start_year, month=1, day=1),
            end_date=pd.Timestamp(year=end_year, month=12, day=31, hour=23, minute=59, second=59),
        )


@dataclass(frozen=True)
class DatasetBundle:
    """Container for the three datasets used in the project."""

    consumption: pd.DataFrame
    weather: pd.DataFrame
    calendar: pd.DataFrame


def _expand_patterns(
    directory: str | Path,
    patterns: Sequence[str],
) -> list[Path]:
    """Return a sorted, deduplicated list of files matching the patterns."""
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")

    discovered: set[Path] = set()

    for pattern in patterns:
        discovered.update(path for path in directory_path.glob(pattern) if path.is_file())

    return sorted(discovered)


def _read_tabular_file(
    path: Path,
    *,
    dtype: str | dict | None = None,
) -> pd.DataFrame:
    """Read CSV or Parquet while preserving source columns."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype=dtype,
            low_memory=False,
        )

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


def _combine_files(
    files: Sequence[Path],
    dataset_name: str,
) -> pd.DataFrame:
    """Read and concatenate a group of files."""
    if not files:
        raise FileNotFoundError(f"No files were found for dataset '{dataset_name}'.")

    frames: list[pd.DataFrame] = []

    for path in files:
        frame = _read_tabular_file(path)
        frame.columns = frame.columns.astype(str).str.strip()
        frame["_SOURCE_PATH"] = str(path)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if combined.empty:
        raise ValueError(f"Dataset '{dataset_name}' contains no records.")

    return combined


def _parse_datetime_column(
    frame: pd.DataFrame,
    column: str,
    *,
    utc: bool = False,
) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(
            f"Required datetime column '{column}' was not found. "
            f"Available columns: {list(frame.columns)}"
        )

    return pd.to_datetime(frame[column], errors="coerce", utc=utc)


def _filter_period(
    frame: pd.DataFrame,
    datetime_column: str,
    period: AnalysisPeriod,
) -> pd.DataFrame:
    """Filter using a timezone-naive comparison series."""
    parsed = _parse_datetime_column(frame, datetime_column)

    # Remove timezone information only for period filtering.
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        comparison = parsed.dt.tz_localize(None)
    else:
        comparison = parsed

    valid_mask = comparison.between(period.start_date, period.end_date, inclusive="both")
    output = frame.loc[valid_mask].copy()
    output[datetime_column] = parsed.loc[valid_mask]

    return output.reset_index(drop=True)


def load_consumption(
    data_directory: str | Path,
    *,
    start_year: int = 2021,
    end_year: int = 2025,
    patterns: Sequence[str] = (
        "hourly_consumption_fsa_*.csv",
        "hourly_consumption_fsa_*.parquet",
    ),
) -> pd.DataFrame:
    """Load and combine all FSA electricity-consumption files."""
    files = _expand_patterns(data_directory, patterns)
    frame = _combine_files(files, "consumption")

    required = {
        "FSA",
        "DATE",
        "HOUR",
        "CUSTOMER_TYPE",
        "PRICE_PLAN",
        "TOTAL_CONSUMPTION",
        "PREMISE_COUNT",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Consumption dataset is missing columns: {missing}")

    for column in ("FSA", "CUSTOMER_TYPE", "PRICE_PLAN"):
        frame[column] = frame[column].astype("string").str.strip()

    for column in ("TOTAL_CONSUMPTION", "PREMISE_COUNT", "HOUR", "SOURCE_PERIOD"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    datetime_column = (
        "TIMESTAMP"
        if "TIMESTAMP" in frame.columns
        else "INTERVAL_START_TIMESTAMP"
    )
    frame[datetime_column] = pd.to_datetime(frame[datetime_column], errors="coerce")

    if "DATE" in frame.columns:
        frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce")

    if "INTERVAL_START_TIMESTAMP" in frame.columns:
        frame["INTERVAL_START_TIMESTAMP"] = pd.to_datetime(
            frame["INTERVAL_START_TIMESTAMP"], errors="coerce"
        )

    if "INTERVAL_END_TIMESTAMP" in frame.columns:
        frame["INTERVAL_END_TIMESTAMP"] = pd.to_datetime(
            frame["INTERVAL_END_TIMESTAMP"], errors="coerce"
        )

    return _filter_period(
        frame,
        datetime_column,
        AnalysisPeriod.from_years(start_year, end_year),
    )


def load_weather(
    data_directory: str | Path,
    *,
    start_year: int = 2021,
    end_year: int = 2025,
    patterns: Sequence[str] = (
        "weather_*.csv",
        "weather_*.parquet",
    ),
) -> pd.DataFrame:
    """Load and combine all station-level hourly ECCC weather files."""
    files = _expand_patterns(data_directory, patterns)
    frame = _combine_files(files, "weather")

    required = {
        "Station Name",
        "Climate ID",
        "Date/Time (LST)",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Weather dataset is missing columns: {missing}")

    frame["Station Name"] = frame["Station Name"].astype("string").str.strip()
    frame["Climate ID"] = pd.to_numeric(frame["Climate ID"], errors="coerce").astype("Int64")
    frame["Date/Time (LST)"] = pd.to_datetime(
        frame["Date/Time (LST)"], errors="coerce"
    )

    numeric_columns = [
        "Longitude (x)",
        "Latitude (y)",
        "Year",
        "Month",
        "Day",
        "Temp (°C)",
        "Dew Point Temp (°C)",
        "Rel Hum (%)",
        "Precip. Amount (mm)",
        "Wind Dir (10s deg)",
        "Wind Spd (km/h)",
        "Visibility (km)",
        "Stn Press (kPa)",
        "Hmdx",
        "Wind Chill",
        "SOURCE_PERIOD",
    ]

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return _filter_period(
        frame,
        "Date/Time (LST)",
        AnalysisPeriod.from_years(start_year, end_year),
    )


def load_calendar(
    data_directory: str | Path,
    *,
    start_year: int = 2021,
    end_year: int = 2025,
    patterns: Sequence[str] = (
        "calendar*.csv",
        "calendar*.parquet",
    ),
) -> pd.DataFrame:
    """Load the project-generated hourly Ontario calendar."""
    files = _expand_patterns(data_directory, patterns)
    frame = _combine_files(files, "calendar")

    if "timestamp_local" not in frame.columns:
        raise ValueError("Calendar dataset must contain 'timestamp_local'.")

    frame["timestamp_local"] = pd.to_datetime(
        frame["timestamp_local"], errors="coerce"
    )

    if "timestamp_utc" in frame.columns:
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"], errors="coerce", utc=True
        )

    if "timestamp_toronto" in frame.columns:
        frame["timestamp_toronto"] = pd.to_datetime(
            frame["timestamp_toronto"], errors="coerce", utc=True
        )

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    return _filter_period(
        frame,
        "timestamp_local",
        AnalysisPeriod.from_years(start_year, end_year),
    )


def load_project_datasets(
    processed_directory: str | Path,
    *,
    start_year: int = 2021,
    end_year: int = 2025,
) -> DatasetBundle:
    """Load all project datasets from the processed data directory."""
    return DatasetBundle(
        consumption=load_consumption(
            processed_directory,
            start_year=start_year,
            end_year=end_year,
        ),
        weather=load_weather(
            processed_directory,
            start_year=start_year,
            end_year=end_year,
        ),
        calendar=load_calendar(
            processed_directory,
            start_year=start_year,
            end_year=end_year,
        ),
    )


def ensure_output_directories(
    project_root: str | Path,
) -> tuple[Path, Path]:
    """Create and return the docs and reports directories."""
    root = Path(project_root)
    docs = root / "docs"
    reports = root / "reports"

    docs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    return docs, reports
