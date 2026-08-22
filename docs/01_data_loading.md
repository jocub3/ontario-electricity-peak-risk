# Data Loading

## Purpose

1. The four raw source files (calendar, two weather stations, six consumption files) are loaded, with no cleaning or joining applied.
2. Each file's shape, column types, and structure are confirmed before the cleaning and join logic is built.

## Design

1. Loading lives in its own module, `src/common/data_loading.py`, separate from cleaning and joining, so each step can be tested on its own.
2. Three functions: `load_calendar()`, `load_weather(station)`, `load_consumption(fsa)`.
3. `load_weather()` renames the timestamp column to `timestamp_local`, the join key used across all datasets.
4. The raw consumption file has one row per (FSA, hour, customer type, price plan). `load_consumption()` aggregates it to one row per FSA-hour by summing `TOTAL_CONSUMPTION` and `PREMISE_COUNT`, both kept as their own columns.
5. `FSA_TO_STATION` maps each of the six FSAs to exactly one weather station, Toronto City or Toronto INTL A. The two stations are never mixed.

## Findings

Findings come from `01_01_data_loading_check.ipynb`, which loads all four files and inspects their shape, columns, and first rows.

1. Calendar and both weather files run through the end of 2026, while all six consumption files stop at the end of 2025. Smart-meter readings have not been produced yet for the later dates. 2021-2025 is the real usable window, the 2026 rows in calendar and weather need to be left out once everything is joined.
2. Row counts in the raw consumption files vary a lot by FSA, from about 101K rows (M5R) to about 216K rows (L4T), even though every FSA covers the same five years of hourly data. This comes from each FSA reporting a different number of CUSTOMER_TYPE and PRICE_PLAN combinations, the same hour repeats once per combination. This confirms the aggregation `load_consumption()` applies is necessary before the data can be used for modeling.
3. All four files loaded with the expected structure, no unexpected errors or missing columns.
