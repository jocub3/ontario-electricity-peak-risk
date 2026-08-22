"""Each raw dataset is cleaned independently, before any joining happens."""

import pandas as pd

# Dropped because each one duplicates information already kept elsewhere in the calendar file
# (e.g. utc_offset_hours is 1-to-1 with is_daylight_saving_time) - not a judgment about their
# relationship with consumption, just internal redundancy.
CALENDAR_COLUMNS_TO_DROP = [
    "timestamp_utc",
    "timestamp_toronto",
    "date",
    "month_name",
    "weekday_name",
    "hour_group",
    "holiday_name",
    "date_index",
    "utc_offset_hours",
]


def clean_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Redundant calendar columns are dropped and the join-key dtype is fixed."""
    df = df.drop(columns=CALENDAR_COLUMNS_TO_DROP)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    return df


# Station metadata (constant within a file), duplicate calendar fields, ingestion metadata, and
# the *Flag columns (essentially 0% populated) are dropped as internal redundancy/no-information.
# Wind Chill and Hmdx are dropped too - both are derived from columns being kept (Temp+Wind Spd,
# Temp+Dew Point/Rel Hum) and are themselves mostly null even within the station that reports them.
# Precip. Amount, Wind Dir, Wind Spd, and Visibility are kept - each is an independent measurement,
# not a duplicate of anything else, so whether they matter for consumption is left for the
# correlation analysis rather than decided here on intuition.
WEATHER_COLUMNS_TO_DROP = [
    "Longitude (x)",
    "Latitude (y)",
    "Station Name",
    "Climate ID",
    "Time (LST)",
    "Year",
    "Month",
    "Day",
    "SOURCE_FILE",
    "SOURCE_PERIOD",
    "Flag",
    "Temp Flag",
    "Dew Point Temp Flag",
    "Rel Hum Flag",
    "Precip. Amount Flag",
    "Wind Dir Flag",
    "Wind Spd Flag",
    "Visibility Flag",
    "Stn Press Flag",
    "Hmdx Flag",
    "Wind Chill Flag",
    "Wind Chill",
    "Hmdx",
    "Weather",
]


# These four are reported by both stations and only have short, scattered gaps (a handful of hours
# at a time) within the 2021-2025 modeling window, so a same-column time interpolation is enough -
# it only looks at the immediate neighboring hours, not a global statistic, so it is safe to do
# here rather than waiting for the train/test fold split.
WEATHER_COLUMNS_TO_INTERPOLATE = [
    "Temp (°C)",
    "Dew Point Temp (°C)",
    "Rel Hum (%)",
    "Stn Press (kPa)",
]


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Redundant and uninformative weather columns are dropped, the join-key dtype is fixed, and short gaps are interpolated."""
    df = df.drop(columns=WEATHER_COLUMNS_TO_DROP)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])

    # limit_area="inside" only fills gaps that have a real value on both sides - it will never
    # extrapolate across the trailing 2026 gap, which has no later value to interpolate towards.
    df[WEATHER_COLUMNS_TO_INTERPOLATE] = df[WEATHER_COLUMNS_TO_INTERPOLATE].interpolate(
        method="linear", limit_area="inside"
    )
    return df


def clean_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """The join-key dtype is fixed for an already-aggregated FSA consumption dataset."""
    df = df.copy()
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    return df
