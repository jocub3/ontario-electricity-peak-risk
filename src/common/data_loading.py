"""Data loading and join utilities.

Loads the three sources under ``data/processed/`` (calendar, weather, and
per-FSA consumption) and joins them into a single hourly dataset, one row
per FSA x hour.

Join key
--------
The three sources share the same fixed-clock (non-DST) local timestamp
column, parsed to identical naive ``YYYY-MM-DD HH:MM:SS`` values, so no
timezone arithmetic is needed (verified against the March/November DST
transition dates):

- ``hourly_consumption_fsa_*.csv``: column ``TIMESTAMP``
- ``weather_toronto_*.csv``: column ``Date/Time (LST)``
- ``calendar_hourly_ontario_timestamp.csv``: column ``timestamp_local``

``timestamp_utc`` and ``timestamp_toronto`` in the calendar file are
DST-aware and kept only for display, never as a join key.

FSA -> weather station assignment
----------------------------------
Each FSA is assigned to its nearest ECCC station, not a blend of both:

- Toronto City   (M5S, M5R, M6G): temperature, humidity, precipitation
- Toronto INTL A (L4T, M9W, M9R): temperature, humidity, wind, visibility

Neither station reports all four variables; the missing ones are left out
rather than filled in from the other station.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
INTERIM_DIR = REPO_ROOT / "data" / "interim"

CALENDAR_FILE = PROCESSED_DIR / "calendar_hourly_ontario_timestamp.csv"

WEATHER_FILES = {
    "toronto_city": PROCESSED_DIR / "weather_toronto_city_6158355.csv",
    "toronto_intl_a": PROCESSED_DIR / "weather_toronto_intl_a_6158731.csv",
}

# Fixed FSA -> weather station assignment.
FSA_STATION_MAP = {
    "M5S": "toronto_city",
    "M5R": "toronto_city",
    "M6G": "toronto_city",
    "L4T": "toronto_intl_a",
    "M9W": "toronto_intl_a",
    "M9R": "toronto_intl_a",
}

CONSUMPTION_FILES = {
    fsa: PROCESSED_DIR / f"hourly_consumption_fsa_{fsa.lower()}.csv"
    for fsa in FSA_STATION_MAP
}

# Renamed to plain, model-friendly column names. A station only appears in
# the output if it actually reports that variable.
WEATHER_COLUMNS = {
    "Temp (°C)": "temperature_c",
    "Rel Hum (%)": "relative_humidity_pct",
    "Precip. Amount (mm)": "precipitation_mm",
    "Wind Spd (km/h)": "wind_speed_kmh",
    "Visibility (km)": "visibility_km",
}

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a source CSV, stripping the UTF-8 BOM present in every file."""
    if not path.exists():
        raise FileNotFoundError(f"Expected data file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def load_calendar() -> pd.DataFrame:
    """Load the hourly calendar table.

    ``timestamp_local`` is the fixed-clock join key. ``timestamp_utc`` and
    ``timestamp_toronto`` are parsed too but are for display only.
    """
    calendar = _read_csv(CALENDAR_FILE)
    calendar["timestamp_local"] = pd.to_datetime(
        calendar["timestamp_local"], format=TIMESTAMP_FORMAT
    )
    calendar["timestamp_utc"] = pd.to_datetime(calendar["timestamp_utc"], utc=True)
    # Left as a plain string: it mixes UTC offsets (-05:00 in winter,
    # -04:00 in summer) by design (DST-aware), which pandas cannot hold in
    # a single non-object datetime dtype. It is for display only, never
    # used as a join key or a feature.
    return calendar


def load_weather(station: str) -> pd.DataFrame:
    """Load one weather station file, keeping only the variables it reports.

    Parameters
    ----------
    station:
        One of ``"toronto_city"`` or ``"toronto_intl_a"``.
    """
    if station not in WEATHER_FILES:
        raise ValueError(f"Unknown station '{station}'. Expected one of {list(WEATHER_FILES)}.")

    weather = _read_csv(WEATHER_FILES[station])
    weather = weather.rename(columns={"Date/Time (LST)": "timestamp_local", **WEATHER_COLUMNS})
    weather["timestamp_local"] = pd.to_datetime(
        weather["timestamp_local"], format=TIMESTAMP_FORMAT
    )

    available_columns = ["timestamp_local"] + [
        column for column in WEATHER_COLUMNS.values() if column in weather.columns
    ]
    weather = weather[available_columns].drop_duplicates(subset="timestamp_local")
    return weather


def load_consumption(fsa: str) -> pd.DataFrame:
    """Load one FSA's consumption file, aggregated to one row per hour.

    The source file has one row per (FSA, TIMESTAMP, CUSTOMER_TYPE,
    PRICE_PLAN); the model target is total consumption per FSA per hour, so
    customer type and price plan are summed away here.
    """
    if fsa not in CONSUMPTION_FILES:
        raise ValueError(f"Unknown FSA '{fsa}'. Expected one of {list(CONSUMPTION_FILES)}.")

    consumption = _read_csv(CONSUMPTION_FILES[fsa])
    consumption["TIMESTAMP"] = pd.to_datetime(consumption["TIMESTAMP"], format=TIMESTAMP_FORMAT)

    grouped = (
        consumption.groupby("TIMESTAMP", as_index=False)
        .agg(
            electricity_consumption=("TOTAL_CONSUMPTION", "sum"),
            premise_count=("PREMISE_COUNT", "sum"),
        )
        .rename(columns={"TIMESTAMP": "timestamp_local"})
    )
    grouped.insert(0, "fsa", fsa)
    return grouped


def build_fsa_hourly_dataset(fsa: str) -> pd.DataFrame:
    """Join calendar + assigned weather station + consumption for one FSA."""
    station = FSA_STATION_MAP[fsa]

    calendar = load_calendar()
    weather = load_weather(station)
    consumption = load_consumption(fsa)

    merged = consumption.merge(
        calendar, on="timestamp_local", how="left", validate="one_to_one"
    )
    merged = merged.merge(
        weather, on="timestamp_local", how="left", validate="one_to_one"
    )
    merged.insert(1, "station", station)

    return merged


def build_all_fsa_dataset() -> pd.DataFrame:
    """Join calendar + weather + consumption for all 6 FSAs, stacked long.

    One row per FSA x hour: the input shape expected by a pooled model
    that uses ``fsa`` as a categorical feature.
    """
    frames = [build_fsa_hourly_dataset(fsa) for fsa in FSA_STATION_MAP]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["fsa", "timestamp_local"]).reset_index(drop=True)
    return combined


def save_interim_dataset(
    df: pd.DataFrame, filename: str = "fsa_hourly_master.parquet"
) -> Path:
    """Save a joined dataset to ``data/interim/`` as parquet."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DIR / filename
    df.to_parquet(output_path, index=False)
    return output_path
