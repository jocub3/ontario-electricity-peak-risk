# Data Quality Report

**Analysis period:** 2021-01-01 through 2025-12-31.

This report profiles the combined consumption, weather, and calendar datasets. It identifies potential issues but does not automatically delete, impute, aggregate, or correct records.

## Dataset overview

| dataset     |    rows |   columns |   memory_mb |   exact_duplicate_rows | minimum_timestamp   | maximum_timestamp   |   missing_timestamp_rows |   source_file_count |
|:------------|--------:|----------:|------------:|-----------------------:|:--------------------|:--------------------|-------------------------:|--------------------:|
| consumption | 1046461 |        11 |     411.566 |                      0 | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                        0 |                   6 |
| weather     |   87648 |        33 |      58.892 |                      0 | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                        0 |                   2 |
| calendar    |   43824 |        45 |      31.956 |                      0 | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                        0 |                   1 |

## Key integrity

| dataset     | expected_key                                 | available_key_columns                        |   missing_key_rows |   duplicate_key_rows |   duplicate_key_groups | key_is_unique   |
|:------------|:---------------------------------------------|:---------------------------------------------|-------------------:|---------------------:|-----------------------:|:----------------|
| consumption | FSA + TIMESTAMP + CUSTOMER_TYPE + PRICE_PLAN | FSA + TIMESTAMP + CUSTOMER_TYPE + PRICE_PLAN |                  0 |                    0 |                      0 | True            |
| weather     | Climate ID + Date/Time (LST)                 | Climate ID + Date/Time (LST)                 |                  0 |                    0 |                      0 | True            |
| calendar    | timestamp_local                              | timestamp_local                              |                  0 |                    0 |                      0 | True            |

### Interpretation note

Consumption records are expected to repeat by FSA and timestamp because the source grain also includes customer type and price plan. Therefore, the detailed consumption key includes those fields.

## Temporal completeness

| dataset     | entity                                           | minimum_timestamp   | maximum_timestamp   |   observed_unique_hours |   expected_continuous_hours |   missing_hour_count |   duplicate_timestamp_rows |   days_with_less_than_24_unique_hours |   days_with_more_than_24_unique_hours |   minimum_hours_in_a_day |   maximum_hours_in_a_day | first_missing_hour_examples   |
|:------------|:-------------------------------------------------|:--------------------|:--------------------|------------------------:|----------------------------:|---------------------:|---------------------------:|--------------------------------------:|--------------------------------------:|-------------------------:|-------------------------:|:------------------------------|
| consumption | FSA=L4T                                          | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                   43824 |                       43824 |                    0 |                     215697 |                                     0 |                                     0 |                       24 |                       24 |                               |
| consumption | FSA=M5R                                          | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                   43824 |                       43824 |                    0 |                     101596 |                                     0 |                                     0 |                       24 |                       24 |                               |
| consumption | FSA=M5S                                          | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                   43824 |                       43824 |                    0 |                     103032 |                                     0 |                                     0 |                       24 |                       24 |                               |
| consumption | FSA=M6G                                          | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                   43824 |                       43824 |                    0 |                     203915 |                                     0 |                                     0 |                       24 |                       24 |                               |
| consumption | FSA=M9R                                          | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                   43824 |                       43824 |                    0 |                     209729 |                                     0 |                                     0 |                       24 |                       24 |                               |
| consumption | FSA=M9W                                          | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                   43824 |                       43824 |                    0 |                     212492 |                                     0 |                                     0 |                       24 |                       24 |                               |
| weather     | Climate ID=6158355 | Station Name=TORONTO CITY   | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                   43824 |                       43824 |                    0 |                          0 |                                     0 |                                     0 |                       24 |                       24 |                               |
| weather     | Climate ID=6158731 | Station Name=TORONTO INTL A | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                   43824 |                       43824 |                    0 |                          0 |                                     0 |                                     0 |                       24 |                       24 |                               |
| calendar    | ALL                                              | 2021-01-01 00:00:00 | 2025-12-31 23:00:00 |                   43824 |                       43824 |                    0 |                          0 |                                     0 |                                     0 |                       24 |                       24 |                               |

## Join readiness against the calendar

| dataset     | entity                                      |   unique_timestamps |   timestamps_missing_from_calendar |   calendar_timestamps_without_entity_data |
|:------------|:--------------------------------------------|--------------------:|-----------------------------------:|------------------------------------------:|
| consumption | FSA=L4T                                     |               43824 |                                  0 |                                         0 |
| consumption | FSA=M5R                                     |               43824 |                                  0 |                                         0 |
| consumption | FSA=M5S                                     |               43824 |                                  0 |                                         0 |
| consumption | FSA=M6G                                     |               43824 |                                  0 |                                         0 |
| consumption | FSA=M9R                                     |               43824 |                                  0 |                                         0 |
| consumption | FSA=M9W                                     |               43824 |                                  0 |                                         0 |
| weather     | Climate ID=6158355 | Station=TORONTO CITY   |               43824 |                                  0 |                                         0 |
| weather     | Climate ID=6158731 | Station=TORONTO INTL A |               43824 |                                  0 |                                         0 |

## Columns with at least 10% missing values

| dataset   | column              | dtype   |   rows |   non_null_count |   missing_count |   missing_pct |   unique_count | constant_column   |
|:----------|:--------------------|:--------|-------:|-----------------:|----------------:|--------------:|---------------:|:------------------|
| calendar  | holiday_name        | object  |  43824 |             1176 |           42648 |       97.3165 |             13 | False             |
| weather   | Flag                | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Temp Flag           | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Precip. Amount Flag | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Wind Dir Flag       | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Wind Spd Flag       | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Stn Press Flag      | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Hmdx Flag           | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Wind Chill Flag     | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Dew Point Temp Flag | object  |  87648 |                1 |           87647 |       99.9989 |              1 | True              |
| weather   | Rel Hum Flag        | object  |  87648 |                1 |           87647 |       99.9989 |              1 | True              |
| weather   | Visibility Flag     | object  |  87648 |                1 |           87647 |       99.9989 |              1 | True              |
| weather   | Wind Chill          | float64 |  87648 |             8782 |           78866 |       89.9804 |             34 | False             |
| weather   | Hmdx                | float64 |  87648 |            15430 |           72218 |       82.3955 |             22 | False             |
| weather   | Weather             | object  |  87648 |            18720 |           68928 |       78.6418 |             70 | False             |
| weather   | Wind Dir (10s deg)  | float64 |  87648 |            43092 |           44556 |       50.8352 |             36 | False             |
| weather   | Precip. Amount (mm) | float64 |  87648 |            43783 |           43865 |       50.0468 |            108 | False             |
| weather   | Visibility (km)     | float64 |  87648 |            43817 |           43831 |       50.008  |             30 | False             |
| weather   | Wind Spd (km/h)     | float64 |  87648 |            43818 |           43830 |       50.0068 |             65 | False             |

Missingness in derived weather measures such as humidex or wind chill may be condition-dependent and should not automatically be interpreted as a data error. Source flags must be reviewed before imputation.

## Constant or single-valued columns

| dataset   | column              | dtype   |   rows |   non_null_count |   missing_count |   missing_pct |   unique_count | constant_column   |
|:----------|:--------------------|:--------|-------:|-----------------:|----------------:|--------------:|---------------:|:------------------|
| weather   | Flag                | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Temp Flag           | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Dew Point Temp Flag | object  |  87648 |                1 |           87647 |       99.9989 |              1 | True              |
| weather   | Rel Hum Flag        | object  |  87648 |                1 |           87647 |       99.9989 |              1 | True              |
| weather   | Precip. Amount Flag | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Wind Dir Flag       | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Wind Spd Flag       | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Visibility Flag     | object  |  87648 |                1 |           87647 |       99.9989 |              1 | True              |
| weather   | Stn Press Flag      | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Hmdx Flag           | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |
| weather   | Wind Chill Flag     | float64 |  87648 |                0 |           87648 |      100      |              0 | True              |

## Required decisions before the Master Dataset

1. Confirm the final timestamp convention used to join consumption, weather, and calendar.
2. Define the aggregation rule from detailed IESO records to one demand value per FSA and hour.
3. Review ECCC flag-code meanings and decide whether flags become quality filters, model features, or documentation fields.
4. Establish variable-specific missing-value rules; do not apply one generic imputation method to all weather fields.
5. Investigate duplicate keys, missing hours, and abnormal values before building modeling features.

Detailed machine-readable results are available in the CSV files under `reports/`.
