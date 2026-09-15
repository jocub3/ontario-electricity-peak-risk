# Data Cleaning Report

## Scope

Input: `E:\jcuenca\OneDrive - GUSCanada\5toTerm\01_Capstone\Codigo\ontario-electricity-peak-risk\data\processed\master_dataset.parquet`  
Output: `E:\jcuenca\OneDrive - GUSCanada\5toTerm\01_Capstone\Codigo\ontario-electricity-peak-risk\data\processed\master_dataset_clean.parquet`

The integrated Master Dataset was preserved. A separate cleaned dataset was
created for EDA and subsequent processing.

## Dataset dimensions

- Before cleaning: **262,944 rows × 71 columns**
- After cleaning: **262,944 rows × 66 columns**

No row was removed as part of routine missing-value or outlier treatment.

## Column actions

| column              | action   | rule                         |   missing_pct_before | reason                                                                                                                 |
|:--------------------|:---------|:-----------------------------|---------------------:|:-----------------------------------------------------------------------------------------------------------------------|
| SOURCE_FILE         | dropped  | drop_if_present              |               0      | Weather source-file traceability is already documented upstream and is not an analytical variable.                     |
| SOURCE_PERIOD       | dropped  | drop_if_present              |               0      | Monthly source-period traceability is already documented upstream and duplicates information available from timestamp. |
| Dew Point Temp Flag | dropped  | drop_if_missing_pct_at_least |              99.9989 | Near-empty ECCC quality flag; not sufficiently populated for analysis.                                                 |
| Rel Hum Flag        | dropped  | drop_if_missing_pct_at_least |              99.9989 | Near-empty ECCC quality flag; not sufficiently populated for analysis.                                                 |
| Visibility Flag     | dropped  | drop_if_missing_pct_at_least |              99.9989 | Near-empty ECCC quality flag; not sufficiently populated for analysis.                                                 |

## Semantic missing-value actions

| column       | replacement   |   values_filled | status   |
|:-------------|:--------------|----------------:|:---------|
| holiday_name | No holiday    |          255888 | filled   |

## Remaining columns with at least 10% missing values

| stage          | column              | dtype    |   rows |   non_null_count |   missing_count |   missing_pct |   unique_count |
|:---------------|:--------------------|:---------|-------:|-----------------:|----------------:|--------------:|---------------:|
| after_cleaning | Wind Chill          | float64  | 262944 |            26346 |          236598 |       89.9804 |             34 |
| after_cleaning | Hmdx                | float64  | 262944 |            46290 |          216654 |       82.3955 |             22 |
| after_cleaning | Weather             | category | 262944 |            56160 |          206784 |       78.6418 |             70 |
| after_cleaning | Wind Dir (10s deg)  | float64  | 262944 |           129276 |          133668 |       50.8352 |             36 |
| after_cleaning | Precip. Amount (mm) | float64  | 262944 |           131349 |          131595 |       50.0468 |            108 |
| after_cleaning | Visibility (km)     | float64  | 262944 |           131451 |          131493 |       50.008  |             30 |
| after_cleaning | Wind Spd (km/h)     | float64  | 262944 |           131454 |          131490 |       50.0068 |             65 |

These weather fields were retained because their missingness may be
condition-dependent or station-dependent. They were not globally imputed.

## Consistency checks

| check                                     |   violations | severity   | status   | description                                                              |
|:------------------------------------------|-------------:|:-----------|:---------|:-------------------------------------------------------------------------|
| required_key_not_null                     |            0 | error      | PASS     | FSA and timestamp must be present for every row.                         |
| unique_fsa_timestamp                      |            0 | error      | PASS     | The cleaned dataset must contain one row per FSA and timestamp.          |
| required_non_null::fsa                    |            0 | error      | PASS     | Required field 'fsa' must not be missing.                                |
| required_non_null::timestamp              |            0 | error      | PASS     | Required field 'timestamp' must not be missing.                          |
| required_non_null::total_consumption_kwh  |            0 | error      | PASS     | Required field 'total_consumption_kwh' must not be missing.              |
| required_non_null::reported_premise_count |            0 | error      | PASS     | Required field 'reported_premise_count' must not be missing.             |
| non_negative_consumption                  |            0 | error      | PASS     | Electricity consumption must not be negative.                            |
| non_negative_premise_count                |            0 | error      | PASS     | Reported premise count must not be negative.                             |
| relative_humidity_range                   |            0 | error      | PASS     | Relative humidity must be between 0 and 100 percent.                     |
| non_negative::Precip. Amount (mm)         |            0 | error      | PASS     | 'Precip. Amount (mm)' must not contain negative values.                  |
| non_negative::Wind Spd (km/h)             |            0 | error      | PASS     | 'Wind Spd (km/h)' must not contain negative values.                      |
| non_negative::Visibility (km)             |            0 | error      | PASS     | 'Visibility (km)' must not contain negative values.                      |
| wind_direction_range                      |            0 | error      | PASS     | Wind direction in tens of degrees must be between 0 and 36.              |
| timestamp_year_consistency                |            0 | error      | PASS     | Calendar year must agree with timestamp.                                 |
| timestamp_month_consistency               |            0 | error      | PASS     | Calendar month must agree with timestamp.                                |
| timestamp_day_consistency                 |            0 | error      | PASS     | Calendar day must agree with timestamp.                                  |
| timestamp_hour_consistency                |            0 | error      | PASS     | Calendar hour must agree with timestamp.                                 |
| binary_domain::is_weekend                 |            0 | error      | PASS     | Binary field 'is_weekend' must contain only 0 or 1.                      |
| binary_domain::is_workday                 |            0 | error      | PASS     | Binary field 'is_workday' must contain only 0 or 1.                      |
| binary_domain::is_monday                  |            0 | error      | PASS     | Binary field 'is_monday' must contain only 0 or 1.                       |
| binary_domain::is_friday                  |            0 | error      | PASS     | Binary field 'is_friday' must contain only 0 or 1.                       |
| binary_domain::is_business_hour           |            0 | error      | PASS     | Binary field 'is_business_hour' must contain only 0 or 1.                |
| binary_domain::is_peak_hour_window        |            0 | error      | PASS     | Binary field 'is_peak_hour_window' must contain only 0 or 1.             |
| binary_domain::is_month_start             |            0 | error      | PASS     | Binary field 'is_month_start' must contain only 0 or 1.                  |
| binary_domain::is_month_end               |            0 | error      | PASS     | Binary field 'is_month_end' must contain only 0 or 1.                    |
| binary_domain::is_year_start              |            0 | error      | PASS     | Binary field 'is_year_start' must contain only 0 or 1.                   |
| binary_domain::is_year_end                |            0 | error      | PASS     | Binary field 'is_year_end' must contain only 0 or 1.                     |
| binary_domain::is_public_holiday          |            0 | error      | PASS     | Binary field 'is_public_holiday' must contain only 0 or 1.               |
| binary_domain::is_day_before_holiday      |            0 | error      | PASS     | Binary field 'is_day_before_holiday' must contain only 0 or 1.           |
| binary_domain::is_day_after_holiday       |            0 | error      | PASS     | Binary field 'is_day_after_holiday' must contain only 0 or 1.            |
| binary_domain::is_long_weekend            |            0 | error      | PASS     | Binary field 'is_long_weekend' must contain only 0 or 1.                 |
| binary_domain::is_daylight_saving_time    |            0 | error      | PASS     | Binary field 'is_daylight_saving_time' must contain only 0 or 1.         |
| binary_domain::is_dst_transition_day      |            0 | error      | PASS     | Binary field 'is_dst_transition_day' must contain only 0 or 1.           |
| binary_domain::is_spring_forward_day      |            0 | error      | PASS     | Binary field 'is_spring_forward_day' must contain only 0 or 1.           |
| binary_domain::is_fall_back_day           |            0 | error      | PASS     | Binary field 'is_fall_back_day' must contain only 0 or 1.                |
| dst_transition_days_have_24_source_hours  |            0 | error      | PASS     | Every FSA must preserve 24 source-aligned hours on DST transition dates. |

## Outlier review

| column                 |   non_null_count |   minimum |     p01 |      q1 |   median |       q3 |      p99 |   maximum |   iqr_lower_bound |   iqr_upper_bound |   iqr_outlier_count |   iqr_outlier_pct | action                    |
|:-----------------------|-----------------:|----------:|--------:|--------:|---------:|---------:|---------:|----------:|------------------:|------------------:|--------------------:|------------------:|:--------------------------|
| total_consumption_kwh  |           262944 |    1377.7 | 2586.33 | 5224.9  |  8111.4  | 11544    | 21178.2  |  33712.2  |          -4253.75 |          21022.6  |                2768 |          1.0527   | reported_only_not_removed |
| reported_premise_count |           262944 |    5803   | 5873    | 6024    |  8674    | 11176    | 12099    |  12120    |          -1704    |          18904    |                   0 |          0        | reported_only_not_removed |
| Temp (°C)              |           262803 |     -21.8 |  -11.9  |    1.9  |    10.1  |    19.2  |    29.4  |     35.8  |            -24.05 |             45.15 |                   0 |          0        | reported_only_not_removed |
| Dew Point Temp (°C)    |           262800 |     -30.1 |  -17.9  |   -3.2  |     3.9  |    12.6  |    21.5  |     26    |            -26.9  |             36.3  |                 108 |          0.041096 | reported_only_not_removed |
| Rel Hum (%)            |           262800 |      14   |   28    |   56    |    68    |    80    |   100    |    100    |             20    |            116    |                 165 |          0.062785 | reported_only_not_removed |
| Precip. Amount (mm)    |           131349 |       0   |    0    |    0    |     0    |     0    |     2.3  |     45    |              0    |              0    |               10791 |          8.21552  | reported_only_not_removed |
| Wind Dir (10s deg)     |           129276 |       1   |    1    |   14    |    24    |    30    |    36    |     36    |            -10    |             54    |                   0 |          0        | reported_only_not_removed |
| Wind Spd (km/h)        |           131454 |       0   |    0    |    9    |    14    |    21    |    43    |     67    |             -9    |             39    |                2520 |          1.91702  | reported_only_not_removed |
| Visibility (km)        |           131451 |       0   |    1.6  |   24.1  |    24.1  |    24.1  |    24.1  |     80.5  |             24.1  |             24.1  |               21933 |         16.6853   | reported_only_not_removed |
| Stn Press (kPa)        |           262803 |      95.7 |   97.7  |   99.31 |    99.88 |   100.43 |   101.72 |    103.25 |             97.63 |            102.11 |                2802 |          1.0662   | reported_only_not_removed |
| Hmdx                   |            46290 |      25   |   25    |   27    |    29    |    32    |    40    |     46    |             19.5  |             39.5  |                 609 |          1.31562  | reported_only_not_removed |
| Wind Chill             |            26346 |     -34   |  -26    |  -14    |   -10    |    -7    |    -2    |     -1    |            -24.5  |              3.5  |                 405 |          1.53723  | reported_only_not_removed |

Outliers were not removed. High electricity demand is analytically meaningful
for both forecasting and Peak-Risk classification. The IQR results are
diagnostic flags, not automatic deletion rules.

## Missing-value policy

- Required keys and the demand target must not be missing.
- `holiday_name` is filled with `No holiday`, because its null value has a clear
  semantic meaning.
- Numeric weather variables remain missing at this phase.
- Any model requiring complete numeric inputs must fit its imputation logic
  using only the training portion of each rolling/expanding validation window.

## Feature-engineering boundary

No lag, rolling, interaction, scaling, encoded, Peak-Risk, or forecast-horizon
variables were created during this phase.
