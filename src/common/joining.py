"""Join the cleaned calendar, weather, and consumption datasets into one FSA-hourly dataset."""

import pandas as pd

from .cleaning import clean_calendar, clean_consumption, clean_weather
from .data_loading import FSA_TO_STATION, load_calendar, load_consumption, load_weather


def build_fsa_hourly_dataset(fsa: str) -> pd.DataFrame:
    """Build the cleaned, joined hourly dataset for a single FSA.

    Consumption only covers 2021-2025, while calendar/weather extend into 2026, so joining
    calendar/weather onto consumption (rather than the other way around) keeps the result
    bounded to the years consumption actually has data for.
    """
    fsa = fsa.upper()
    consumption = clean_consumption(load_consumption(fsa))
    calendar = clean_calendar(load_calendar())
    weather = clean_weather(load_weather(FSA_TO_STATION[fsa]))

    merged = consumption.merge(calendar, on="timestamp_local", how="left", validate="one_to_one")
    merged = merged.merge(weather, on="timestamp_local", how="left", validate="one_to_one")
    return merged


def build_all_fsa_dataset() -> pd.DataFrame:
    """Build the pooled, cleaned hourly dataset for all 6 FSAs."""
    return pd.concat(
        [build_fsa_hourly_dataset(fsa) for fsa in FSA_TO_STATION],
        ignore_index=True,
    )
