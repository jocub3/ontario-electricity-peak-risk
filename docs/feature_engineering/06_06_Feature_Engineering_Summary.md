# Feature Engineering Summary
    
    ## Feature families

    | feature_family              |   feature_count |
|:----------------------------|----------------:|
| administrative_or_redundant |               2 |
| existing                    |              47 |
| interaction                 |               2 |
| operational_exogenous       |               1 |
| spatial                     |               1 |
| target                      |               1 |
| target_lag                  |               6 |
| target_rolling              |              16 |
| temporal_cyclical           |               8 |
| traceability                |               8 |
| weather_change              |               4 |
| weather_nonlinear           |               2 |
| weather_rolling             |               4 |

    ## Preliminary feature priorities

    | preliminary_priority   |   feature_count |
|:-----------------------|----------------:|
| administrative         |               8 |
| high                   |              41 |
| low                    |               2 |
| medium                 |              48 |
| target                 |               1 |

    ## Validation

    | check                                                             |   violations | status   | description                                                                 |
|:------------------------------------------------------------------|-------------:|:---------|:----------------------------------------------------------------------------|
| row_count_preserved                                               |            0 | PASS     | Feature Engineering must not silently add or remove rows.                   |
| unique_key                                                        |            0 | PASS     | FSA + timestamp must remain unique.                                         |
| key_not_null                                                      |            0 | PASS     | Key columns must not contain null values.                                   |
| no_infinite_values                                                |            0 | PASS     | Numeric engineered features must not contain positive or negative infinity. |
| lag_1h_alignment                                                  |            0 | PASS     | One-hour target lag must equal the prior FSA observation.                   |
| rolling_history_required::total_consumption_kwh_rolling_mean_3h   |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_std_3h    |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_min_3h    |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_max_3h    |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_mean_6h   |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_std_6h    |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_min_6h    |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_max_6h    |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_mean_24h  |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_std_24h   |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_min_24h   |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_max_24h   |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_mean_168h |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_std_168h  |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_min_168h  |            0 | PASS     | Rolling target features must require historical observations.               |
| rolling_history_required::total_consumption_kwh_rolling_max_168h  |            0 | PASS     | Rolling target features must require historical observations.               |

    ## Modeling handoff

    `feature_dataset.parquet` is the static feature table produced by the Feature Engineering Phase.

    Modeling code must still:

    1. perform chronological train/validation/test splits;
    2. fit missing-value treatment using training data only;
    3. create the final Peak-Risk thresholds and labels inside each training window;
    4. ensure that future weather inputs correspond to forecast-available weather,
    not future observed weather;
    5. select final predictors separately for each candidate model.

    ## Forecast-Horizon Constraint

    Historical demand features must be reviewed for each forecast horizon before model training. 
    Their presence in `feature_dataset.parquet` does not imply that they are directly available for all 24 future hours.
    