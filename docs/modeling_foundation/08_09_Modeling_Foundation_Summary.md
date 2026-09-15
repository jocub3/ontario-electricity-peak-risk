# Modeling Foundation Summary

## Common Horizon-Safe Temporal Splits

| fold      | role       | configured_train_start   | configured_train_end   | safe_train_origin_end   |   train_rows | train_first_origin   | train_last_origin   |   train_fsas | configured_evaluation_start   | configured_evaluation_end   | safe_evaluation_origin_end   |   evaluation_rows | evaluation_first_origin   | evaluation_last_origin   |   evaluation_fsas |   max_horizon_hours |
|:----------|:-----------|:-------------------------|:-----------------------|:------------------------|-------------:|:---------------------|:--------------------|-------------:|:------------------------------|:----------------------------|:-----------------------------|------------------:|:--------------------------|:-------------------------|------------------:|--------------------:|
| fold_2023 | validation | 2021-01-01 00:00:00      | 2022-12-31 23:00:00    | 2022-12-30 23:00:00     |       104976 | 2021-01-01 00:00:00  | 2022-12-30 23:00:00 |            6 | 2023-01-01 00:00:00           | 2023-12-31 23:00:00         | 2023-12-30 23:00:00          |             52416 | 2023-01-01 00:00:00       | 2023-12-30 23:00:00      |                 6 |                  24 |
| fold_2024 | validation | 2021-01-01 00:00:00      | 2023-12-31 23:00:00    | 2023-12-30 23:00:00     |       157536 | 2021-01-01 00:00:00  | 2023-12-30 23:00:00 |            6 | 2024-01-01 00:00:00           | 2024-12-31 23:00:00         | 2024-12-30 23:00:00          |             52560 | 2024-01-01 00:00:00       | 2024-12-30 23:00:00      |                 6 |                  24 |
| test_2025 | test       | 2021-01-01 00:00:00      | 2024-12-31 23:00:00    | 2024-12-30 23:00:00     |       210240 | 2021-01-01 00:00:00  | 2024-12-30 23:00:00 |            6 | 2025-01-01 00:00:00           | 2025-12-31 23:00:00         | 2025-12-30 23:00:00          |             52416 | 2025-01-01 00:00:00       | 2025-12-30 23:00:00      |                 6 |                  24 |

## FSA Coverage by Fold

| fold      | role       | fsa   |   train_rows |   evaluation_rows | status   |
|:----------|:-----------|:------|-------------:|------------------:|:---------|
| fold_2023 | validation | L4T   |        17496 |              8736 | PASS     |
| fold_2023 | validation | M5R   |        17496 |              8736 | PASS     |
| fold_2023 | validation | M5S   |        17496 |              8736 | PASS     |
| fold_2023 | validation | M6G   |        17496 |              8736 | PASS     |
| fold_2023 | validation | M9R   |        17496 |              8736 | PASS     |
| fold_2023 | validation | M9W   |        17496 |              8736 | PASS     |
| fold_2024 | validation | L4T   |        26256 |              8760 | PASS     |
| fold_2024 | validation | M5R   |        26256 |              8760 | PASS     |
| fold_2024 | validation | M5S   |        26256 |              8760 | PASS     |
| fold_2024 | validation | M6G   |        26256 |              8760 | PASS     |
| fold_2024 | validation | M9R   |        26256 |              8760 | PASS     |
| fold_2024 | validation | M9W   |        26256 |              8760 | PASS     |
| test_2025 | test       | L4T   |        35040 |              8736 | PASS     |
| test_2025 | test       | M5R   |        35040 |              8736 | PASS     |
| test_2025 | test       | M5S   |        35040 |              8736 | PASS     |
| test_2025 | test       | M6G   |        35040 |              8736 | PASS     |
| test_2025 | test       | M9R   |        35040 |              8736 | PASS     |
| test_2025 | test       | M9W   |        35040 |              8736 | PASS     |

## Official Project Metrics

| task        | metric            | role      | direction        | description                                            |
|:------------|:------------------|:----------|:-----------------|:-------------------------------------------------------|
| forecasting | MAE               | primary   | lower_is_better  | Mean absolute forecast error in kWh.                   |
| forecasting | RMSE              | primary   | lower_is_better  | Root mean squared forecast error in kWh.               |
| forecasting | MAPE              | primary   | lower_is_better  | Mean absolute percentage error for non-zero demand.    |
| forecasting | WAPE              | secondary | lower_is_better  | Absolute error relative to total observed demand.      |
| forecasting | Bias              | secondary | closer_to_zero   | Mean signed forecast error.                            |
| peak_risk   | Precision         | primary   | higher_is_better | Share of predicted Peaks that are true Peaks.          |
| peak_risk   | Recall            | primary   | higher_is_better | Share of true Peaks correctly detected.                |
| peak_risk   | F1                | primary   | higher_is_better | Harmonic mean of Precision and Recall.                 |
| peak_risk   | PR-AUC            | primary   | higher_is_better | Precision-Recall area; emphasized for class imbalance. |
| peak_risk   | ROC-AUC           | secondary | higher_is_better | Area under the ROC curve.                              |
| peak_risk   | Balanced Accuracy | secondary | higher_is_better | Average recall across both classes.                    |

## Peak-Risk Multi-Horizon Policy

|   horizon | target   |   target_offset_hours | definition                                                           | threshold_scope          |
|----------:|:---------|----------------------:|:---------------------------------------------------------------------|:-------------------------|
|         1 | peak_h01 |                     1 | Peak status of the corresponding FSA at forecast_origin + 1 hour(s)  | training_only_fsa_season |
|         2 | peak_h02 |                     2 | Peak status of the corresponding FSA at forecast_origin + 2 hour(s)  | training_only_fsa_season |
|         3 | peak_h03 |                     3 | Peak status of the corresponding FSA at forecast_origin + 3 hour(s)  | training_only_fsa_season |
|         4 | peak_h04 |                     4 | Peak status of the corresponding FSA at forecast_origin + 4 hour(s)  | training_only_fsa_season |
|         5 | peak_h05 |                     5 | Peak status of the corresponding FSA at forecast_origin + 5 hour(s)  | training_only_fsa_season |
|         6 | peak_h06 |                     6 | Peak status of the corresponding FSA at forecast_origin + 6 hour(s)  | training_only_fsa_season |
|         7 | peak_h07 |                     7 | Peak status of the corresponding FSA at forecast_origin + 7 hour(s)  | training_only_fsa_season |
|         8 | peak_h08 |                     8 | Peak status of the corresponding FSA at forecast_origin + 8 hour(s)  | training_only_fsa_season |
|         9 | peak_h09 |                     9 | Peak status of the corresponding FSA at forecast_origin + 9 hour(s)  | training_only_fsa_season |
|        10 | peak_h10 |                    10 | Peak status of the corresponding FSA at forecast_origin + 10 hour(s) | training_only_fsa_season |
|        11 | peak_h11 |                    11 | Peak status of the corresponding FSA at forecast_origin + 11 hour(s) | training_only_fsa_season |
|        12 | peak_h12 |                    12 | Peak status of the corresponding FSA at forecast_origin + 12 hour(s) | training_only_fsa_season |
|        13 | peak_h13 |                    13 | Peak status of the corresponding FSA at forecast_origin + 13 hour(s) | training_only_fsa_season |
|        14 | peak_h14 |                    14 | Peak status of the corresponding FSA at forecast_origin + 14 hour(s) | training_only_fsa_season |
|        15 | peak_h15 |                    15 | Peak status of the corresponding FSA at forecast_origin + 15 hour(s) | training_only_fsa_season |
|        16 | peak_h16 |                    16 | Peak status of the corresponding FSA at forecast_origin + 16 hour(s) | training_only_fsa_season |
|        17 | peak_h17 |                    17 | Peak status of the corresponding FSA at forecast_origin + 17 hour(s) | training_only_fsa_season |
|        18 | peak_h18 |                    18 | Peak status of the corresponding FSA at forecast_origin + 18 hour(s) | training_only_fsa_season |
|        19 | peak_h19 |                    19 | Peak status of the corresponding FSA at forecast_origin + 19 hour(s) | training_only_fsa_season |
|        20 | peak_h20 |                    20 | Peak status of the corresponding FSA at forecast_origin + 20 hour(s) | training_only_fsa_season |
|        21 | peak_h21 |                    21 | Peak status of the corresponding FSA at forecast_origin + 21 hour(s) | training_only_fsa_season |
|        22 | peak_h22 |                    22 | Peak status of the corresponding FSA at forecast_origin + 22 hour(s) | training_only_fsa_season |
|        23 | peak_h23 |                    23 | Peak status of the corresponding FSA at forecast_origin + 23 hour(s) | training_only_fsa_season |
|        24 | peak_h24 |                    24 | Peak status of the corresponding FSA at forecast_origin + 24 hour(s) | training_only_fsa_season |

## Framework Validation

| fold      | role            | check                                      | status   |   details | latest_validation_end   | test_start          | configured_value   |
|:----------|:----------------|:-------------------------------------------|:---------|----------:|:------------------------|:--------------------|:-------------------|
| fold_2023 | validation      | non_empty_train                            | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2023 | validation      | non_empty_evaluation                       | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2023 | validation      | train_precedes_evaluation                  | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2023 | validation      | same_fsa_coverage                          | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2023 | validation      | no_temporal_overlap                        | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2023 | validation      | train_origin_is_horizon_safe               | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2023 | validation      | evaluation_origin_is_horizon_safe          | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2024 | validation      | non_empty_train                            | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2024 | validation      | non_empty_evaluation                       | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2024 | validation      | train_precedes_evaluation                  | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2024 | validation      | same_fsa_coverage                          | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2024 | validation      | no_temporal_overlap                        | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2024 | validation      | train_origin_is_horizon_safe               | PASS     |       nan | NaT                     | NaT                 | nan                |
| fold_2024 | validation      | evaluation_origin_is_horizon_safe          | PASS     |       nan | NaT                     | NaT                 | nan                |
| test_2025 | test            | non_empty_train                            | PASS     |       nan | NaT                     | NaT                 | nan                |
| test_2025 | test            | non_empty_evaluation                       | PASS     |       nan | NaT                     | NaT                 | nan                |
| test_2025 | test            | train_precedes_evaluation                  | PASS     |       nan | NaT                     | NaT                 | nan                |
| test_2025 | test            | same_fsa_coverage                          | PASS     |       nan | NaT                     | NaT                 | nan                |
| test_2025 | test            | no_temporal_overlap                        | PASS     |       nan | NaT                     | NaT                 | nan                |
| test_2025 | test            | train_origin_is_horizon_safe               | PASS     |       nan | NaT                     | NaT                 | nan                |
| test_2025 | test            | evaluation_origin_is_horizon_safe          | PASS     |       nan | NaT                     | NaT                 | nan                |
| global    | forecast_target | all_forecasting_target_columns_available   | PASS     |           | NaT                     | NaT                 | nan                |
| fold_2023 | validation      | train_horizon_complete                     | PASS     |           | NaT                     | NaT                 | nan                |
| fold_2023 | validation      | evaluation_horizon_complete                | PASS     |           | NaT                     | NaT                 | nan                |
| fold_2023 | validation      | train_h24_within_training_period           | PASS     |           | NaT                     | NaT                 | nan                |
| fold_2023 | validation      | evaluation_h24_within_evaluation_period    | PASS     |           | NaT                     | NaT                 | nan                |
| fold_2024 | validation      | train_horizon_complete                     | PASS     |           | NaT                     | NaT                 | nan                |
| fold_2024 | validation      | evaluation_horizon_complete                | PASS     |           | NaT                     | NaT                 | nan                |
| fold_2024 | validation      | train_h24_within_training_period           | PASS     |           | NaT                     | NaT                 | nan                |
| fold_2024 | validation      | evaluation_h24_within_evaluation_period    | PASS     |           | NaT                     | NaT                 | nan                |
| test_2025 | test            | train_horizon_complete                     | PASS     |           | NaT                     | NaT                 | nan                |
| test_2025 | test            | evaluation_horizon_complete                | PASS     |           | NaT                     | NaT                 | nan                |
| test_2025 | test            | train_h24_within_training_period           | PASS     |           | NaT                     | NaT                 | nan                |
| test_2025 | test            | evaluation_h24_within_evaluation_period    | PASS     |           | NaT                     | NaT                 | nan                |
| nan       | nan             | final_holdout_after_all_validation_periods | PASS     |       nan | 2024-12-31 23:00:00     | 2025-01-01 00:00:00 | nan                |
| nan       | nan             | preprocessing_fit_scope_training_only      | PASS     |       nan | NaT                     | NaT                 | training_only      |
| nan       | nan             | forecasting_selection_metric_is_primary    | PASS     |       nan | NaT                     | NaT                 | nan                |
| nan       | nan             | peak_selection_metric_is_primary           | PASS     |       nan | NaT                     | NaT                 | nan                |
| nan       | nan             | classification_probability_threshold_valid | PASS     |       nan | NaT                     | NaT                 | nan                |
| nan       | nan             | forecast_and_peak_horizons_match           | PASS     |       nan | NaT                     | NaT                 | nan                |

## Handoff to Model Branches

After this framework is reviewed and merged into `develop`, individual model branches should be created from the updated `develop` branch.

The first model branches will be created separately:

- `forecasting/seasonal-naive-baseline`
- `peak-risk/logistic-regression-baseline`

Neither baseline is implemented in the Modeling Foundation Phase's `modeling/framework` branch.

All future model branches must reuse the common:

- temporal folds;
- horizon-safe forecast-origin boundaries;
- target definitions;
- evaluation metrics;
- model-selection metrics;
- Peak-Risk threshold policy;
- classification comparison threshold;
- preprocessing and anti-leakage rules.
