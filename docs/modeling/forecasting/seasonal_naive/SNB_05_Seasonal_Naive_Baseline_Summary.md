# Daily Seasonal Naive Forecasting Baseline

## Baseline definition

For each future target hour, the model predicts electricity demand using the
observed demand from the same hour 24 hours earlier.

No learned preprocessing, feature scaling, parameter fitting, or hyperparameter
tuning is used.

## Development policy

Only the shared validation folds are evaluated in this branch.

The final 2025 holdout is not evaluated and remains protected for the final
model-selection stage.

## Fold metrics

| fold      |     mae |     rmse |    mape |    wape |      bias |   n_observations |
|:----------|--------:|---------:|--------:|--------:|----------:|-----------------:|
| fold_2023 | 603.791 |  980.377 | 6.50976 | 6.69051 | -2.41133  |          1257984 |
| fold_2024 | 620.283 | 1007.98  | 6.66932 | 6.76163 |  0.731706 |          1261440 |

## Pooled development metrics

| model                |     mae |    rmse |    mape |    wape |      bias |   n_observations |
|:---------------------|--------:|--------:|--------:|--------:|----------:|-----------------:|
| seasonal_naive_daily | 612.049 | 994.293 | 6.58965 | 6.72641 | -0.837658 |          2519424 |

## Prediction coverage

| fold      | target     |   available_predictions |   missing_predictions |   coverage_pct | status   |
|:----------|:-----------|------------------------:|----------------------:|---------------:|:---------|
| fold_2023 | target_h01 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h02 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h03 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h04 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h05 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h06 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h07 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h08 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h09 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h10 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h11 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h12 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h13 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h14 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h15 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h16 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h17 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h18 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h19 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h20 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h21 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h22 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h23 |                   52416 |                     0 |            100 | PASS     |
| fold_2023 | target_h24 |                   52416 |                     0 |            100 | PASS     |
| fold_2024 | target_h01 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h02 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h03 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h04 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h05 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h06 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h07 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h08 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h09 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h10 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h11 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h12 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h13 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h14 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h15 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h16 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h17 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h18 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h19 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h20 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h21 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h22 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h23 |                   52560 |                     0 |            100 | PASS     |
| fold_2024 | target_h24 |                   52560 |                     0 |            100 | PASS     |

## Interpretation

This model is the Forecasting reference benchmark. Candidate Forecasting models
must demonstrate improvement relative to this baseline under the same common
splits, horizons, and evaluation metrics.
