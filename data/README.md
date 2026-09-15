# Data directory

Large datasets must not be committed to GitHub.

## Folders

- `raw/`: original files downloaded from IESO, ECCC, and other official sources.
- `interim/`: partially cleaned or merged files.
- `processed/`: modeling-ready datasets.
- `external/`: supplementary reference files, such as FSA boundaries.
- `sample/`: small anonymized or reduced files that may be committed for testing.



## Raw Data

The four files under `data/raw/` are the raw data, and they have the following details:

| File | Rows | Range | Role |
|---|---|---|---|
| `calendar_hourly_ontario.csv` | 52,585 | 2021-01-01 → 2027-01-01 | Calendar features key |
| `weather_toronto_city_6158355.csv` | 52,585 | 2021-01-01 → 2026-12-31 | Downtown station |
| `weather_toronto_intl_a_6158731.csv` | 52,585 | 2021-01-01 → 2026-12-31 | Airport station (Pearson) |
| `hourly_consumption_selected_fsa_all.csv` | 1,046,462 | 2021-01-01 → 2025-12-31 | IESO consumption for 6 FSAs; `HOUR` (1-24, hour-ending) |

## FSA → region → station mapping

Each FSA is assigned to its **nearest** weather station. Stations are not merged or averaged, since averaging would smooth out the temperature/precipitation extremes that peak-demand events tend to correlate with.

| Region | FSAs | Station | Available weather variables |
|---|---|---|---|
| `Downtown` | M5S, M5R, M6G | Toronto City (6158355) | temperature, relative humidity, precipitation |
| `Airport-West` | L4T, M9W, M9R | Toronto Pearson / INTL A (6158731) | temperature, relative humidity, wind speed, visibility |

Each region's target is the hourly sum of `TOTAL_CONSUMPTION` across its 3 FSAs.


