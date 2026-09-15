# Seasonal Naive Forecasting Baseline – Overview

## Purpose

The Seasonal Naive model establishes the reference performance for the electricity-demand Forecasting task.

Its purpose is not to produce the most accurate forecast possible. Instead, it provides a simple and transparent benchmark that the candidate Forecasting models must improve upon under the same validation folds, forecast horizons, and evaluation metrics.

## Conceptual Approach

The model follows a simple rule:

> For each future hour, use the electricity consumption observed at the same hour 24 hours earlier.

For example, the predicted demand for Tuesday at 3:00 PM is based on the observed demand for Monday at 3:00 PM.

The model produces predictions for all 24 forecast horizons:

`target_h01 ... target_h24`

Because the method does not learn coefficients or model parameters, it does not require a training process, feature scaling, feature selection, or hyperparameter tuning.

## Evaluation Process

The baseline uses the common Modeling Foundation rules.

Two development folds are evaluated:

- **Fold 2023:** historical information is used to evaluate forecasts during 2023.
- **Fold 2024:** the expanded historical period is used to evaluate forecasts during 2024.

The final 2025 holdout is not evaluated at this stage.

Performance is reviewed:

- globally;
- by validation fold;
- by FSA;
- by forecast horizon.

## Main Results

### Overall Development Performance

| Metric | Result |
|---|---:|
| MAE | 612.05 kWh |
| RMSE | 994.29 kWh |
| MAPE | 6.59% |
| WAPE | 6.73% |
| Bias | -0.84 kWh |

### Performance by Validation Fold

| Fold | MAE (kWh) | RMSE (kWh) | MAPE |
|---|---:|---:|---:|
| 2023 | 603.79 | 980.38 | 6.51% |
| 2024 | 620.28 | 1,007.98 | 6.67% |

The results are relatively stable between the two validation periods.

### Performance by FSA

The absolute forecasting error differs across FSAs.

- **M5S** shows the lowest absolute MAE, approximately 238–241 kWh.
- **L4T** and **M9W** show the largest absolute MAE, approximately 841–879 kWh.
- Percentage-based metrics should also be considered when comparing FSAs because the areas operate at different electricity-demand scales.

### Performance by Forecast Horizon

MAE remains almost unchanged from `h+1` through `h+24`.

This is expected for this baseline because each horizon independently uses the corresponding hour from the previous day. The method does not recursively feed its own previous predictions into later horizons.

### Prediction Coverage

Prediction coverage is **100% for all 24 horizons in both validation folds**.

No missing Seasonal Naive predictions were detected.

## Interpretation

### Technical Interpretation

The Seasonal Naive benchmark achieves an overall MAE of approximately **612 kWh** and MAPE of approximately **6.59%**.

The similar results between the 2023 and 2024 folds indicate that the benchmark is reasonably stable across the two development periods. Its near-zero overall Bias also indicates no substantial systematic tendency to over-forecast or under-forecast demand.

### Non-Technical Interpretation

A very simple rule, using the same hour from the previous day, already provides a reasonably strong forecast.

Therefore, more advanced models should not be considered successful simply because they can generate predictions. They should demonstrate a measurable improvement over this benchmark.

The key comparison for future models will be:

> Can a more sophisticated model reduce the forecasting error below the Seasonal Naive reference while remaining stable across FSAs, years, and the full 24-hour horizon?

## Conclusion

**Accepted as the Forecasting baseline.**

No tuning is recommended for this model. Its value is precisely its simplicity, transparency, and reproducibility.

Candidate Forecasting models should be compared against this benchmark using the same Modeling Foundation rules.


