"""Load raw calendar, weather, and consumption data. No cleaning, no joining - that happens later."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

CALENDAR_FILE = DATA_DIR / "calendar_hourly_ontario_timestamp.csv"

STATION_FILES = {
    "city": DATA_DIR / "weather_toronto_city_6158355.csv",
    "intl_a": DATA_DIR / "weather_toronto_intl_a_6158731.csv",
}

# Each FSA is served by exactly one weather station - the two stations must never be mixed.
FSA_TO_STATION = {
    "M5S": "city",
    "M5R": "city",
    "M6G": "city",
    "L4T": "intl_a",
    "M9W": "intl_a",
    "M9R": "intl_a",
}


def load_calendar() -> pd.DataFrame:
    """Load the raw calendar file as-is."""
    return pd.read_csv(CALENDAR_FILE)


def load_weather(station: str) -> pd.DataFrame:
    """Load one raw weather station file, with its timestamp column renamed to match the join key."""
    df = pd.read_csv(STATION_FILES[station], low_memory=False)
    return df.rename(columns={"Date/Time (LST)": "timestamp_local"})


def load_consumption(fsa: str) -> pd.DataFrame:
    """Load one FSA's consumption file and collapse it to one row per FSA-hour.

    The raw file has one row per (FSA, hour, customer_type, price_plan). CUSTOMER_TYPE and
    PRICE_PLAN are dropped by summing TOTAL_CONSUMPTION and PREMISE_COUNT across them - both
    stay in the result as their own columns, just no longer split by customer segment.
    """
    df = pd.read_csv(DATA_DIR / f"hourly_consumption_fsa_{fsa.lower()}.csv")
    return (
        df.groupby(["FSA", "TIMESTAMP"], as_index=False)[["TOTAL_CONSUMPTION", "PREMISE_COUNT"]]
        .sum()
        .rename(columns={"TIMESTAMP": "timestamp_local"})
    )
