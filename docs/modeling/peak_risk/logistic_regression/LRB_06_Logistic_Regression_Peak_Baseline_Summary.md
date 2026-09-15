# Logistic Regression Peak-Risk Baseline

## Baseline definition

The baseline fits one Logistic Regression classifier for each forecast horizon
from `peak_h01` through `peak_h24`.

The model uses a deliberately small, interpretable predictor set:

- FSA;
- target season;
- cyclic target-hour, weekday, and month representations;
- weekend status;
- same target hour's electricity demand 24 hours earlier;
- same target hour's electricity demand 168 hours earlier.

No observed future electricity demand is used as a predictor.

## Preprocessing

Numeric features use the shared Logistic Regression preprocessing profile:
training-only imputation and standardization.

Categorical variables use training-only imputation and one-hot encoding.

No hyperparameter search is performed. This is a fixed baseline configuration.

## Peak target policy

Peak thresholds are fitted independently inside each training fold using the
shared 97.5th-percentile FSA + season policy.

The frozen training thresholds are then applied to validation observations.

## Development policy

Only the 2023 and 2024 validation folds are evaluated.

The final 2025 holdout remains untouched.

## Fold metrics

| fold      |   precision |   recall |       f1 |   balanced_accuracy |   n_observations |   positive_rate_pct |   pr_auc |   roc_auc |
|:----------|------------:|---------:|---------:|--------------------:|-----------------:|--------------------:|---------:|----------:|
| fold_2023 |    0.898704 | 0.321783 | 0.473888 |            0.658658 |          1257984 |            10.9652  | 0.744155 |  0.941553 |
| fold_2024 |    0.700125 | 0.26479  | 0.384253 |            0.630026 |          1261440 |             4.00939 | 0.538786 |  0.95454  |

## Pooled development metrics

| model                             |   precision |   recall |       f1 |   balanced_accuracy |   n_observations |   positive_rate_pct |   pr_auc |   roc_auc |
|:----------------------------------|------------:|---------:|---------:|--------------------:|-----------------:|--------------------:|---------:|----------:|
| logistic_regression_peak_baseline |    0.843267 | 0.306492 | 0.449581 |            0.650943 |          2519424 |             7.48254 | 0.675276 |  0.943635 |

## Class balance

| fold      |   horizon |   observations |   peak_hours |   peak_rate_pct |
|:----------|----------:|---------------:|-------------:|----------------:|
| fold_2023 |         1 |          52416 |         5743 |        10.9566  |
| fold_2023 |         2 |          52416 |         5743 |        10.9566  |
| fold_2023 |         3 |          52416 |         5743 |        10.9566  |
| fold_2023 |         4 |          52416 |         5743 |        10.9566  |
| fold_2023 |         5 |          52416 |         5743 |        10.9566  |
| fold_2023 |         6 |          52416 |         5743 |        10.9566  |
| fold_2023 |         7 |          52416 |         5743 |        10.9566  |
| fold_2023 |         8 |          52416 |         5743 |        10.9566  |
| fold_2023 |         9 |          52416 |         5743 |        10.9566  |
| fold_2023 |        10 |          52416 |         5743 |        10.9566  |
| fold_2023 |        11 |          52416 |         5743 |        10.9566  |
| fold_2023 |        12 |          52416 |         5744 |        10.9585  |
| fold_2023 |        13 |          52416 |         5745 |        10.9604  |
| fold_2023 |        14 |          52416 |         5746 |        10.9623  |
| fold_2023 |        15 |          52416 |         5747 |        10.9642  |
| fold_2023 |        16 |          52416 |         5748 |        10.9661  |
| fold_2023 |        17 |          52416 |         5750 |        10.9699  |
| fold_2023 |        18 |          52416 |         5752 |        10.9737  |
| fold_2023 |        19 |          52416 |         5754 |        10.9776  |
| fold_2023 |        20 |          52416 |         5755 |        10.9795  |
| fold_2023 |        21 |          52416 |         5756 |        10.9814  |
| fold_2023 |        22 |          52416 |         5757 |        10.9833  |
| fold_2023 |        23 |          52416 |         5757 |        10.9833  |
| fold_2023 |        24 |          52416 |         5757 |        10.9833  |
| fold_2024 |         1 |          52560 |         2107 |         4.00875 |
| fold_2024 |         2 |          52560 |         2107 |         4.00875 |
| fold_2024 |         3 |          52560 |         2107 |         4.00875 |
| fold_2024 |         4 |          52560 |         2107 |         4.00875 |
| fold_2024 |         5 |          52560 |         2107 |         4.00875 |
| fold_2024 |         6 |          52560 |         2107 |         4.00875 |
| fold_2024 |         7 |          52560 |         2107 |         4.00875 |
| fold_2024 |         8 |          52560 |         2107 |         4.00875 |
| fold_2024 |         9 |          52560 |         2107 |         4.00875 |
| fold_2024 |        10 |          52560 |         2107 |         4.00875 |
| fold_2024 |        11 |          52560 |         2107 |         4.00875 |
| fold_2024 |        12 |          52560 |         2107 |         4.00875 |
| fold_2024 |        13 |          52560 |         2107 |         4.00875 |
| fold_2024 |        14 |          52560 |         2107 |         4.00875 |
| fold_2024 |        15 |          52560 |         2107 |         4.00875 |
| fold_2024 |        16 |          52560 |         2107 |         4.00875 |
| fold_2024 |        17 |          52560 |         2108 |         4.01065 |
| fold_2024 |        18 |          52560 |         2108 |         4.01065 |
| fold_2024 |        19 |          52560 |         2108 |         4.01065 |
| fold_2024 |        20 |          52560 |         2108 |         4.01065 |
| fold_2024 |        21 |          52560 |         2108 |         4.01065 |
| fold_2024 |        22 |          52560 |         2108 |         4.01065 |
| fold_2024 |        23 |          52560 |         2108 |         4.01065 |
| fold_2024 |        24 |          52560 |         2108 |         4.01065 |

## Interpretation

This model is the classification reference benchmark. Candidate Peak-Risk
models must demonstrate improvement under the same folds, target methodology,
probability threshold, and official evaluation metrics.
