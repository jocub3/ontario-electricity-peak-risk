# Cleaning and Joining

## Purpose

1. Each raw dataset (calendar, weather, consumption) is cleaned on its own, before any joining happens.
2. The three cleaned datasets are joined into one pooled hourly dataset, one row per FSA-hour.

## Design

1. `src/common/cleaning.py` holds one function per dataset: `clean_calendar()`, `clean_weather()`, `clean_consumption()`. Each fixes the `timestamp_local` dtype from text to `datetime64`, the join key used across all three datasets.
2. `clean_calendar()` drops 9 redundant columns (`timestamp_utc`, `timestamp_toronto`, `date`, `month_name`, `weekday_name`, `hour_group`, `holiday_name`, `date_index`, `utc_offset_hours`), each a duplicate of another column being kept. DST flags and cyclical encodings (`hour_sin`/`cos`, `weekday_sin`/`cos`, `month_sin`/`cos`) are kept, since whether they help predict consumption can only be judged later, once calendar is joined with consumption.
3. `clean_weather()` drops station metadata, calendar/timestamp duplicates, and columns with no real information (every `*Flag` column, `Wind Chill`, `Hmdx`, `Weather`). Eight measurement columns are kept: `Temp (°C)`, `Dew Point Temp (°C)`, `Rel Hum (%)`, `Precip. Amount (mm)`, `Wind Dir (10s deg)`, `Wind Spd (km/h)`, `Visibility (km)`, `Stn Press (kPa)`.
4. Four of those columns (`Temp`, `Dew Point Temp`, `Rel Hum`, `Stn Press`) are linearly interpolated with `limit_area="inside"`, which only fills a gap that has a real value on both sides. It never extrapolates past the last known value.
5. `clean_consumption()` only fixes the timestamp dtype. The row-per-FSA-hour aggregation already happens in `load_consumption()`, so there is nothing left to drop or fill.
6. `src/common/joining.py` builds the final dataset. `build_fsa_hourly_dataset(fsa)` merges the cleaned consumption table with calendar and with the correct weather station (`FSA_TO_STATION`), both merges on `timestamp_local` with `validate="one_to_one"`. `build_all_fsa_dataset()` concatenates all 6 FSAs into the pooled dataset saved as `fsa_hourly_master.parquet`.

## Findings

Findings come from `01_02_data_cleaning.ipynb`, which runs each cleaning function and the join, and inspects the result at every step.

1. Calendar and both weather files cover 2021 through the end of 2026, while all six consumption files stop at the end of 2025, since smart-meter readings have not been produced yet for 2026. The join is what actually removes 2026 from the result: consumption is used as the left side of both merges, so calendar and weather rows for 2026 simply have no matching consumption row and never appear in the joined output. No explicit date filter is applied, the exclusion is a direct consequence of which table drives the join.
2. Because of that same 2021-2026 span, checking null percentages on the raw weather files before joining is misleading. `Temp`, `Dew Point Temp`, `Rel Hum`, and `Stn Press` looked about 7% null in both stations across the full file, but almost all of that is one unbroken gap from 2026-07 to 2026-12, future data that simply does not exist yet, not a real measurement gap.
3. Restricted to the real 2021-2025 window, the true gap in those four columns is small: 0.09% in Toronto City (at most 6 consecutive hours) and 0.01% in Toronto INTL A (single isolated hours). This small, short gap is exactly what the linear interpolation in `clean_weather()` fills, using only the immediate neighboring hours, not a global statistic, so it carries no train/test leakage risk.
4. After joining all 6 FSAs (262,944 rows, 47 columns), only four columns still have nulls, and they are the four weather measurements each station simply does not report: `Wind Dir (10s deg)` (50.8%), `Precip. Amount (mm)` (50.0%), `Visibility (km)` (50.0%), `Wind Spd (km/h)` (50.0%). This is expected given `FSA_TO_STATION`, not a gap to fill, since one station reports each of these and the other never does.
