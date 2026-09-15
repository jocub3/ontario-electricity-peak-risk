# SARIMAX Model Overview

## 1. Purpose

The SARIMAX model was developed as the statistical time-series candidate within the electricity demand forecasting framework.

Its purpose is to forecast hourly electricity consumption for the next 24 hours for each Forward Sortation Area (FSA), using historical demand patterns and the temporal structure present in the data.

Unlike the Seasonal Naive baseline, which reproduces historical demand patterns directly, SARIMAX explicitly models temporal dependence and seasonality. This provides a statistically grounded forecasting approach that can be compared with the Machine Learning models developed in the project.

---

## 2. Model Concept

SARIMAX stands for **Seasonal AutoRegressive Integrated Moving Average with eXogenous variables**.

Conceptually, the model predicts future electricity demand by learning how current demand is related to:

- previous demand observations;
- previous forecasting errors;
- recurring temporal patterns;
- seasonal behavior;
- and, when applicable, external explanatory variables.

For this project, the most important idea is that electricity demand is not independent from one hour to the next. Recent consumption and recurring daily patterns contain information that can help predict future demand.

SARIMAX provides a statistical framework for representing these temporal relationships.

---

## 3. Why SARIMAX Was Included

SARIMAX was selected because electricity demand is inherently a time-series problem.

The model provides several advantages for the project:

- it explicitly represents temporal dependence;
- it can capture recurring seasonal patterns;
- it provides a statistical alternative to Machine Learning models;
- it allows forecasting performance to be evaluated across the complete 24-hour horizon;
- and its residuals can be analyzed to identify temporal patterns that remain unexplained.

Including SARIMAX also increases methodological diversity in the forecasting comparison.

The final candidate set therefore contains fundamentally different forecasting approaches:

- Seasonal Naive as the simple temporal baseline;
- SARIMAX as the statistical time-series model;
- Random Forest as a tree-based ensemble model;
- XGBoost as a gradient-boosting model;
- LightGBM as an efficient gradient-boosting model.

---

## 4. Forecasting Strategy

The model follows the common Modeling Foundation established for the project.

Forecasts are generated for:

> **h+1, h+2, ..., h+24**

where each horizon represents one future hour.

The evaluation uses the same temporal folds, FSA structure, targets, and official forecasting metrics used by the other candidate models.

The final 2025 holdout period remains excluded from model development and validation.

This ensures that SARIMAX can later be compared fairly against the other forecasting models.

---

## 5. Rolling-Origin Implementation

The initial SARIMAX implementation was computationally expensive because repeated model fitting produced very long execution times and convergence warnings.

A revised **rolling-origin implementation** was therefore introduced.

Instead of repeatedly estimating the complete statistical model for every forecasting origin, the revised process fits the model for each FSA and temporal fold and then performs rolling forecasting while updating the model state as new observations become available.

This preserves the temporal forecasting logic while making the experiment computationally feasible.

A smoke test was performed before the complete execution to confirm that the revised process worked correctly.

The final execution successfully completed all FSA × fold tasks.

---

## 6. Final Model Specification

The selected SARIMAX specification was:

- **ARIMA order:** `(2, 0, 1)`
- **Seasonal order:** `(1, 0, 0, 24)`
- **Trend:** constant

The seasonal period of 24 represents the recurring daily structure of hourly electricity demand. :contentReference[oaicite:0]{index=0}

The specification was selected through the model-selection stage before the final rolling-origin evaluation.

A deliberately limited model-selection strategy was used rather than an exhaustive search because SARIMAX estimation is substantially more computationally expensive than the Machine Learning candidates.

---

## 7. Training and Validation

The final rolling implementation processed **12 FSA × fold tasks**.

All 12 model fits converged successfully, with no convergence warnings during the final execution.

The average fitting time was approximately **102 seconds**, while the slowest fit required approximately **158 seconds**.

The validation process generated approximately **2.52 million predictions** across FSAs, temporal folds, and forecast horizons.

These results confirmed that the revised implementation solved the major computational and convergence problems observed in the original SARIMAX execution.

---

# 8. Model Results

## 8.1 Global Performance

The aggregated validation results were:

| Metric | Result |
|---|---:|
| MAE | 583.64 kWh |
| RMSE | 929.49 kWh |
| MAPE | 6.09% |
| WAPE | 6.41% |
| Bias | -164.62 kWh |

Overall, SARIMAX produced a **MAPE of approximately 6.1%** across the complete validation experiment.

In practical terms, this means that the forecasts differed from the observed electricity demand by approximately 6% on average.

The negative global bias indicates a moderate overall tendency toward overprediction, although this behavior is not consistent across all FSAs.

---

## 8.2 Performance Across Validation Folds

Performance remained relatively stable between the two validation periods.

| Fold | MAE | MAPE | WAPE |
|---|---:|---:|---:|
| 2023 | 598.34 kWh | 6.19% | 6.63% |
| 2024 | 568.97 kWh | 5.99% | 6.20% |

The model performed slightly better in the 2024 validation fold.

More importantly, no major deterioration was observed between the two periods.

### Interpretation

This suggests that SARIMAX demonstrates reasonable temporal stability.

The model does not appear to perform well during one validation period and fail completely during another, which is an important characteristic for a forecasting system intended to operate over time.

---

## 8.3 Performance by Forecast Horizon

One of the clearest results is the relationship between forecasting horizon and prediction error.

For example, during the 2024 validation period:

| Horizon | MAE | MAPE |
|---|---:|---:|
| h+1 | ~134 kWh | ~1.52% |
| h+2 | ~242 kWh | ~2.68% |
| h+3 | ~338 kWh | ~3.69% |
| h+6 | ~524 kWh | ~5.65% |
| h+12 | ~606 kWh | ~6.29% |
| h+18 | ~673 kWh | ~7.05% |
| h+24 | ~696 kWh | ~7.25% |

Forecasting accuracy decreases progressively as the prediction horizon increases.

This behavior is expected because uncertainty accumulates as the model attempts to predict further into the future.

### Interpretation

SARIMAX performs particularly well for very short-term forecasting.

The approximately **1.5% MAPE at h+1** indicates that the immediate next-hour forecast is considerably more accurate than the complete 24-hour forecast.

This distinction is important for the final model comparison because a model that is not the overall winner may still be the strongest candidate for specific forecast horizons.

---

## 8.4 Performance by FSA

Forecasting difficulty varies considerably between FSAs.

During the 2023 validation period, representative results included:

| FSA | MAE | MAPE |
|---|---:|---:|
| M5S | ~202 kWh | ~5.05% |
| M5R | ~409 kWh | ~5.42% |
| M9R | ~422 kWh | ~7.22% |
| M6G | ~594 kWh | ~5.01% |
| L4T | ~739 kWh | ~6.33% |
| M9W | ~1,224 kWh | ~8.13% |

The results demonstrate that forecasting performance depends substantially on the characteristics of each FSA.

M5S produced comparatively small absolute errors, while M9W was substantially more difficult to forecast.

This confirms that a single global performance metric cannot fully describe model behavior.

---

# 9. M9W-2023 Structural Change

The most important FSA-specific finding occurred for **M9W during 2023**.

The approximate results were:

| Measure | M9W-2023 |
|---|---:|
| Actual mean demand | 14,259 kWh |
| Predicted mean demand | 13,196 kWh |
| MAE | 1,224 kWh |
| MAPE | 8.13% |
| Mean residual | +1,064 kWh |

Unlike the moderate global tendency toward overprediction, SARIMAX strongly **underpredicted M9W during 2023**.

The average residual exceeded 1,000 kWh.

For comparison, the average MAE for the other FSAs during the same analysis was approximately 473 kWh, with a mean residual close to zero.

This behavior is consistent with the structural change previously identified during the Target Definition phase.

Electricity demand in M9W increased substantially around the 2022–2023 transition, together with a major change in reported premise count.

Because SARIMAX depends strongly on historical temporal patterns, it could not fully anticipate this change using the preceding historical behavior.

Performance improved during the following validation period:

- M9W 2023 MAPE: approximately **8.13%**
- M9W 2024 MAPE: approximately **6.14%**

This improvement is consistent with the model having access to the changed 2023 demand regime when forecasting the subsequent period.

### Interpretation

M9W illustrates an important limitation of purely historical time-series forecasting.

When the underlying demand-generating process changes substantially, historical relationships may temporarily become less representative of future behavior.

This finding should therefore be considered during the final model comparison and interpretation.

---

# 10. Weather-Related Error Analysis

Forecasting errors were also evaluated across temperature ranges.

Representative MAE values were:

| Temperature Range | MAE |
|---|---:|
| 0–10°C | ~465 kWh |
| -10–0°C | ~546 kWh |
| 10–20°C | ~561 kWh |
| < -10°C | ~717 kWh |
| 20–25°C | ~800 kWh |
| 25–30°C | ~911 kWh |
| > 30°C | ~996 kWh |

The lowest errors occurred under moderate temperature conditions.

Errors increased during temperature extremes, particularly during hot weather.

### Interpretation

This analysis does not establish that temperature directly causes the forecasting errors.

However, it shows that extreme weather conditions are associated with more difficult forecasting conditions.

This finding supports the project's decision to include weather-related information in the forecasting framework and provides an important point of comparison with Machine Learning models that may capture nonlinear relationships between weather and electricity demand.

---

# 11. Residual Diagnostics

Several residual analyses were performed to evaluate what information remained unexplained by the model.

The diagnostics included:

- residual distribution;
- residual versus predicted values;
- residual evolution over time;
- monthly residual behavior;
- Q-Q analysis;
- residual autocorrelation;
- FSA-specific residual analysis;
- M9W residual timeline.

The residuals do not indicate that the SARIMAX model is invalid, but they show that some systematic structure remains unexplained.

This is particularly visible around difficult periods and FSAs such as M9W-2023.

### Interpretation

A strong forecasting model should leave relatively little predictable structure in its errors.

The SARIMAX diagnostics suggest that the model captures an important part of the temporal behavior of electricity demand, but not all of it.

This provides an opportunity for the Machine Learning candidates to improve forecasting performance by capturing nonlinear relationships and interactions that SARIMAX may not represent as effectively.

---

# 12. Strengths

The main strengths identified during the experiment are:

- statistically grounded time-series methodology;
- explicit representation of temporal dependence;
- daily seasonal structure;
- strong short-horizon forecasting performance;
- stable results across the 2023 and 2024 validation folds;
- successful convergence across all final FSA × fold fits;
- interpretable forecast-horizon degradation;
- useful residual diagnostics;
- methodological diversity relative to the Machine Learning candidates.

The model therefore provides a meaningful statistical benchmark beyond the simple Seasonal Naive baseline.

---

# 13. Limitations

The main limitations are:

- forecasting accuracy progressively decreases toward h+24;
- performance varies substantially between FSAs;
- structural changes such as M9W-2023 are difficult to anticipate from historical temporal patterns;
- extreme temperature conditions are associated with larger errors;
- residual diagnostics indicate that some temporal structure remains unexplained;
- SARIMAX estimation is computationally more expensive than several alternative approaches;
- model specification and tuning must therefore remain reasonably constrained.

These limitations do not invalidate the model. Instead, they identify situations in which alternative forecasting methods may provide additional value.

---

# 14. Hyperparameter Tuning Decision

No additional extensive SARIMAX tuning was performed after the final rolling-origin evaluation.

This was a deliberate modeling decision.

The selected specification:

> **SARIMAX (2,0,1) × (1,0,0,24), with constant trend**

successfully converged, produced stable validation results, and provided reasonable forecasting accuracy.

A broader search over autoregressive, moving-average, seasonal, trend, and exogenous configurations would significantly increase computational cost.

Additional tuning will therefore only be reconsidered after the complete forecasting model comparison.

If SARIMAX is close to the best-performing candidate, a limited refinement may be justified. If another model clearly outperforms SARIMAX, extensive additional tuning would provide limited practical value.

The final 2025 holdout remains excluded from this decision.

---

# 15. Overall Interpretation

SARIMAX successfully fulfilled its role as the project's statistical time-series forecasting candidate.

The model achieved approximately:

> **6.09% global validation MAPE**

while demonstrating particularly strong performance for short forecasting horizons.

The results also revealed two important characteristics of the forecasting problem:

1. prediction uncertainty increases progressively from h+1 toward h+24; and
2. structural demand changes, such as the one observed in M9W around 2022–2023, can significantly reduce the effectiveness of models that rely strongly on historical temporal relationships.

These findings are useful beyond the performance of SARIMAX itself because they provide additional understanding of the electricity-demand forecasting problem.

---

# 16. Conclusion

The final SARIMAX implementation is considered a **valid candidate model for the forecasting comparison stage**.

The model:

- completed all required validation folds;
- successfully converged;
- generated forecasts across the full 24-hour horizon;
- produced stable results across validation periods;
- achieved reasonable overall forecasting accuracy;
- and revealed interpretable limitations related to forecast horizon, FSA behavior, structural change, and extreme weather conditions.

No additional tuning is considered necessary at this stage.

SARIMAX should therefore be preserved in its current configuration and compared under the common evaluation framework against Seasonal Naive, Random Forest, XGBoost, and LightGBM.

The final model selection should not rely exclusively on one global metric. Performance by forecast horizon, FSA, temporal stability, computational requirements, and behavior under difficult conditions should also be considered before selecting the final forecasting approach.