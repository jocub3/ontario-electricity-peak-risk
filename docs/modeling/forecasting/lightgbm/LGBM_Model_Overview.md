# LightGBM Regressor — Forecasting Candidate Model Overview

## 1. Purpose

The **LightGBM Regressor** was developed as a candidate model for the 24-hour electricity-demand forecasting task.

Its objective is to estimate electricity consumption independently for each forecast horizon from **h+1 through h+24**, while following the same Modeling Foundation used by the other forecasting candidates.

The model is evaluated under the common expanding-window validation framework, with results reported globally and by validation fold, FSA, forecast horizon, month, and selected stress cases.

The final **2025 holdout was not evaluated** during this stage.

---

## 2. Conceptual Modeling Process

The LightGBM forecasting process can be understood as follows:

1. Historical electricity demand and contextual information available at the forecast origin are prepared.
2. The model uses recent demand behavior, calendar context, geographic information, and available weather conditions to learn relationships associated with future electricity consumption.
3. A separate regression model is developed for each forecast horizon from `h+1` through `h+24`.
4. A moderate hyperparameter search is performed using development folds only.
5. The selected configuration is evaluated over the complete set of 24 horizons.
6. Performance is reviewed globally and by fold, FSA, horizon, month, and selected difficult cases.
7. Saved validation predictions are used for additional diagnostics without retraining the model.

LightGBM is a gradient-boosted decision-tree model. Conceptually, it builds a sequence of trees in which later trees improve the errors left by earlier trees. It is designed to model nonlinear relationships and interactions efficiently on large tabular datasets.

---

## 3. Hyperparameter Selection

A moderate tuning process compared several LightGBM configurations using representative forecast horizons and the common development folds.

The selected configuration was:

| Parameter | Selected value |
|---|---:|
| Number of estimators | 400 |
| Number of leaves | 63 |
| Learning rate | 0.05 |
| Maximum depth | -1 |
| Minimum child samples | 30 |
| Subsample | 0.80 |
| Column sample by tree | 0.80 |
| L2 regularization | 0.50 |

The tuning results showed:

| Configuration | Mean MAE |
|---:|---:|
| **2 — selected** | **553.61 kWh** |
| 3 | 556.94 kWh |
| 4 | 565.03 kWh |
| 1 | 567.27 kWh |

The difference between the two best configurations is small. Therefore, the current tuning is considered sufficient for candidate-model evaluation and no additional tuning is performed at this stage.

---

## 4. Overall Validation Results

Across all development validation predictions, LightGBM obtained:

| Metric | Result |
|---|---:|
| MAE | **580.84 kWh** |
| RMSE | **1,044.37 kWh** |
| MAPE | **5.74%** |
| WAPE | **6.38%** |
| Bias | **-213.04 kWh** |
| Validation predictions | **2,519,424** |

### Technical Interpretation

The model produces an average absolute forecasting error of approximately 581 kWh.

MAPE is approximately 5.74%, indicating an average absolute percentage error below 6%.

The negative official Bias indicates an overall tendency to **underpredict** electricity demand under the project convention:

`Bias = Predicted - Actual`

### Non-Technical Interpretation

LightGBM generally tracks electricity consumption reasonably well, but its accuracy changes depending on the validation year, FSA, season, temperature conditions, and forecast distance.

The global result is also affected by a particularly difficult case involving M9W during 2023.

---

## 5. Validation Fold Performance

| Fold | MAE (kWh) | RMSE (kWh) | MAPE | WAPE | Bias (kWh) |
|---|---:|---:|---:|---:|---:|
| 2023 | **691.05** | 1,242.05 | 6.51% | 7.66% | -382.75 |
| 2024 | **470.92** | 799.98 | 4.96% | 5.13% | -43.80 |

The model performs substantially better in 2024.

MAE decreases from approximately **691 kWh to 471 kWh**, while systematic underprediction also becomes much smaller.

### Interpretation

The difference between validation folds indicates sensitivity to changes in the underlying demand regime.

The improvement in 2024 is consistent with the expanding-window validation design because additional historical information is available for training. However, a large part of the 2023 deterioration is concentrated in M9W.

---

## 6. Performance Across FSAs

### Fold 2023

| FSA | MAE (kWh) | MAPE | Bias (kWh) |
|---|---:|---:|---:|
| L4T | 613.52 | 5.00% | -163.02 |
| M5R | 402.51 | 5.25% | 53.83 |
| M5S | 191.42 | 4.73% | 3.62 |
| M6G | 644.14 | 5.18% | -388.50 |
| M9R | 366.85 | 6.09% | 83.39 |
| **M9W** | **1,927.86** | **12.81%** | **-1,885.79** |

### Fold 2024

| FSA | MAE (kWh) | MAPE | Bias (kWh) |
|---|---:|---:|---:|
| L4T | 615.17 | 4.87% | -118.72 |
| M5R | 381.40 | 5.02% | 5.27 |
| M5S | 193.26 | 4.85% | 37.48 |
| M6G | 569.33 | 4.72% | -119.95 |
| M9R | 349.38 | 5.65% | -52.10 |
| **M9W** | **717.00** | **4.68%** | **-14.79** |

Five FSAs show relatively stable percentage errors. M9W-2023 is clearly different from the remaining observations and remains the principal spatial-temporal stress case.

---

## 7. M9W-2023 Stress Case

A dedicated diagnostic analysis was performed because M9W had a substantially larger 2023 forecasting error.

For M9W-2023:

| Metric | Result |
|---|---:|
| Observations | 209,664 |
| Mean actual demand | 14,259.20 kWh |
| Mean predicted demand | 12,373.41 kWh |
| MAE | **1,927.86 kWh** |
| MAPE | **12.81%** |
| Mean diagnostic residual | **+1,885.79 kWh** |
| Median diagnostic residual | 1,501.43 kWh |
| P95 absolute error | 5,232.17 kWh |
| Maximum absolute error | 14,501.00 kWh |

The additional diagnostic analysis uses:

`Residual = Actual - Predicted`

Therefore, positive diagnostic residuals indicate **underprediction**.

This is consistent with the official Bias of `-1,885.79 kWh`, because the project Bias convention uses:

`Bias = Predicted - Actual`

Both measures describe the same behavior: LightGBM substantially underpredicts M9W in 2023.

### M9W versus the other FSAs

| Group | MAE | MAPE | Mean diagnostic residual |
|---|---:|---:|---:|
| **M9W** | **1,927.86 kWh** | **12.81%** | **+1,885.79 kWh** |
| Other five FSAs | **443.69 kWh** | **5.25%** | **+82.14 kWh** |

The M9W error is therefore more than four times the average absolute error of the other five FSAs.

### Interpretation

The poor 2023 global performance is not a uniform failure of LightGBM across the complete dataset.

A substantial portion of the deterioration is concentrated in M9W, where the model systematically predicts a lower demand level than was observed.

This is treated as a genuine temporal-generalization stress case rather than evidence of an implementation error.

---

## 8. M9W Monthly Behavior

The monthly M9W analysis shows that underprediction is already present at the beginning of 2023 and becomes considerably larger during late spring and summer.

| Month | Actual Mean | Predicted Mean | Difference | Difference % |
|---|---:|---:|---:|---:|
| January | 14,738.86 | 13,566.53 | 1,172.33 | 7.95% |
| February | 14,916.54 | 13,538.13 | 1,378.41 | 9.24% |
| March | 14,345.51 | 13,027.46 | 1,318.05 | 9.19% |
| April | 13,025.59 | 11,566.69 | 1,458.90 | 11.20% |
| May | 12,669.60 | 10,807.43 | 1,862.16 | 14.70% |
| June | 14,427.74 | 11,853.26 | 2,574.48 | 17.84% |
| **July** | **16,919.12** | **13,598.50** | **3,320.62** | **19.63%** |
| August | 14,915.90 | 12,232.09 | 2,683.81 | 17.99% |
| September | 13,939.56 | 11,263.90 | 2,675.65 | 19.19% |
| October | 12,769.97 | 11,198.92 | 1,571.05 | 12.30% |
| November | 13,886.06 | 12,481.37 | 1,404.69 | 10.12% |
| December | 14,573.02 | 13,417.94 | 1,155.08 | 7.93% |

### Interpretation

There is no single isolated failure point.

M9W begins 2023 at a demand level that LightGBM already underestimates, and the difference increases considerably from approximately **May through September**.

July is the most difficult month in absolute terms.

The Actual-vs-Predicted timeline confirms that LightGBM follows part of the overall demand shape but fails to reproduce the full higher demand level and several extreme periods.

---

## 9. Monthly and Seasonal Error Patterns

The validation results reveal a consistent broader seasonal pattern beyond M9W.

### Fold 2023

| Month | MAE |
|---|---:|
| January | 469.80 kWh |
| May | 552.68 kWh |
| June | 900.83 kWh |
| **July** | **1,326.93 kWh** |
| August | 1,019.78 kWh |
| September | 945.20 kWh |
| December | 426.72 kWh |

### Fold 2024

| Month | MAE |
|---|---:|
| January | 330.65 kWh |
| May | 326.66 kWh |
| June | 834.21 kWh |
| **July** | **935.78 kWh** |
| August | 823.47 kWh |
| September | 532.99 kWh |
| November | 243.94 kWh |
| December | 343.19 kWh |

### Interpretation

Two effects are visible:

1. **A general summer increase in forecasting difficulty**, visible in both validation years.
2. **An exceptional M9W-2023 level mismatch**, which increases the 2023 error further.

This distinction is important because the summer error pattern persists even after the M9W problem becomes much smaller in 2024.

---

## 10. Forecast Error by Temperature Band

A dedicated diagnostic compared model error with the temperature observed at the forecast origin.

| Temperature Band | Observations | MAE | MAPE |
|---|---:|---:|---:|
| < -10°C | 23,112 | 572.08 kWh | 4.74% |
| -10 to 0°C | 329,904 | 442.73 kWh | 4.26% |
| 0 to 10°C | 882,936 | **391.43 kWh** | **4.32%** |
| 10 to 20°C | 745,992 | 573.94 kWh | 6.07% |
| 20 to 25°C | 410,688 | 933.25 kWh | 8.39% |
| 25 to 30°C | 118,296 | 1,125.39 kWh | 8.88% |
| **> 30°C** | 7,416 | **1,709.58 kWh** | **10.80%** |

### Interpretation

The relationship between forecast error and temperature is not uniform.

The lowest errors occur in moderate/cool conditions, particularly around 0–10°C. Error increases substantially as temperature rises above 20°C and becomes particularly large above 30°C.

This finding is consistent with the higher forecasting errors observed during summer.

However, this diagnostic does **not** prove that temperature causes the model errors. High temperatures may coincide with more volatile or extreme demand regimes and other seasonal effects.

It does show that **high-temperature periods are an important forecasting stress condition** and should be considered during final model comparison.

---

## 11. Forecast-Horizon Behavior

Forecasting accuracy generally decreases as the prediction horizon increases.

### Fold 2024 — selected horizons

| Horizon | MAE | MAPE |
|---|---:|---:|
| h+1 | **289.35 kWh** | **3.21%** |
| h+6 | 401.30 kWh | 4.23% |
| h+12 | 491.71 kWh | 5.13% |
| h+18 | 539.84 kWh | 5.66% |
| h+24 | **567.79 kWh** | **5.98%** |

The FSA × Horizon heatmaps show that this degradation occurs across the geographic areas rather than being limited to one FSA.

M9W-2023 remains difficult across multiple forecast horizons, indicating that its problem is not simply a long-horizon forecasting issue.

### Interpretation

Near-term electricity demand is easier to forecast than demand approaching 24 hours ahead.

This is expected in multi-horizon forecasting and reinforces the importance of comparing models separately by horizon.

---

## 12. Predictor Importance

The LightGBM feature-importance output indicates that several types of information are actively used by the model, including:

- temperature;
- rolling demand summaries;
- reported premise count;
- station pressure;
- historical demand lags;
- weekday/calendar information;
- humidity;
- FSA-related information.

The LightGBM importance output is based on the model's tree-importance mechanism and should **not** be interpreted directly as percentage contribution or causality.

### Interpretation

Unlike a model whose importance is overwhelmingly dominated by one or two historical lags, LightGBM appears to distribute predictive decisions across a broader combination of historical demand, weather, calendar, and contextual information.

Temperature is particularly notable because the separate temperature-band diagnostic also shows much larger errors under high-temperature conditions.

If LightGBM becomes a finalist, SHAP-based analysis would provide a more reliable and detailed interpretation of feature contribution and direction.

---

## 13. Comparison with Existing Forecasting Models

Current pooled validation results provide the following context:

| Model | MAE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|
| Seasonal Naive baseline | 612.05 | 994.29 | 6.59% | 6.73% |
| Random Forest Regressor | **542.26** | **926.18** | **5.55%** | **5.96%** |
| XGBoost Regressor | 606.27 | 1,106.77 | 5.93% | 6.66% |
| LightGBM Regressor | **580.84** | 1,044.37 | **5.74%** | **6.38%** |

LightGBM:

- improves on XGBoost in MAE, RMSE, MAPE, and WAPE;
- improves on the Seasonal Naive baseline in MAE, MAPE, and WAPE;
- currently remains behind Random Forest on the principal pooled forecasting metrics.

An especially relevant comparison is the difficult 2023 fold:

- XGBoost MAE: approximately **741.73 kWh**
- LightGBM MAE: approximately **691.05 kWh**

In 2024 the two boosting models are essentially tied in MAE:

- XGBoost: approximately **471.18 kWh**
- LightGBM: approximately **470.92 kWh**

### Interpretation

The principal advantage of LightGBM over XGBoost appears to come from handling the more difficult 2023 period somewhat better.

However, Random Forest currently remains the leading candidate among the evaluated machine-learning forecasting models on pooled validation accuracy.

---

## 14. Visual Diagnostics

The LightGBM evaluation includes complementary diagnostic visualizations:

1. MAE by forecast horizon.
2. Actual vs Predicted scatter plot.
3. Residual distribution.
4. Full validation Actual vs Predicted timeline.
5. Monthly MAE by fold.
6. Monthly mean residual by fold.
7. FSA × Month MAE heatmaps.
8. FSA × Fold MAE heatmap.
9. FSA × Horizon MAE heatmaps.
10. M9W monthly MAE.
11. M9W monthly residual.
12. M9W Actual vs Predicted timeline.
13. M9W residual timeline.
14. M9W Actual vs Predicted monthly level.
15. Forecast error by temperature band.

Together, these plots show that error is not randomly distributed. It is concentrated in identifiable stress conditions:

- later forecast horizons;
- summer months;
- high-temperature periods;
- M9W during 2023.

---

## 15. Main Strengths

The LightGBM candidate demonstrates several positive characteristics:

- It successfully forecasts all 24 horizons.
- It respects the common Modeling Foundation and anti-leakage rules.
- It performs better globally than XGBoost on the current validation metrics.
- It improves substantially in the 2024 validation fold.
- Percentage error is reasonably consistent across most FSAs.
- It captures nonlinear relationships and interactions efficiently.
- It uses a broader combination of historical, weather, temporal, and contextual predictors.
- Its horizon degradation is realistic and interpretable.
- The model provides useful standardized outputs for later model comparison.
- It handles M9W-2023 better than XGBoost, although the case remains difficult.

---

## 16. Main Limitations and Observations

The evaluation identifies several important limitations:

1. Random Forest currently outperforms LightGBM on the main pooled forecasting metrics.
2. M9W-2023 remains a major source of error.
3. The model systematically underpredicts M9W during 2023.
4. Summer months are consistently more difficult in both validation folds.
5. Forecast errors rise sharply under high-temperature conditions.
6. Forecast accuracy decreases as the prediction horizon becomes longer.
7. Extreme errors continue to influence RMSE.
8. The current tree feature-importance measure does not provide causal or directional interpretation.
9. The two best tuning configurations are very close, suggesting that additional tuning may provide only modest gains.

These observations do not indicate an implementation problem. They define the conditions under which LightGBM should be compared against the other forecasting candidates.

---

## 17. Current Model Decision

**Status: Accepted as a valid Forecasting candidate model.**

No additional hyperparameter tuning is performed at this stage.

The current LightGBM configuration is sufficiently developed for model comparison. Additional tuning should be reconsidered only if LightGBM becomes the best or a near-best candidate after all forecasting models have been evaluated.

Potential future refinements could include:

- a more focused hyperparameter search;
- horizon-group-specific parameter configurations;
- SHAP-based interpretation;
- additional robustness work for structural demand changes;
- explicit use of operational future weather forecasts when available.

The final **2025 holdout remains protected** and has not been used for candidate-model selection.

---

## 18. Modeling Implications

The LightGBM experiment provides several useful conclusions for the remaining forecasting workflow:

- Historical electricity-demand behavior remains essential for short-term forecasting.
- Weather and contextual information appear to contribute more broadly to LightGBM's decision structure than in some other tree models.
- High-temperature periods are an important forecast stress condition.
- Summer error should be explicitly compared across all final candidate models.
- M9W should remain a specific temporal-generalization stress test.
- A strong final model should reduce M9W-2023 error without sacrificing performance in the other FSAs.
- Short- and long-horizon accuracy should continue to be evaluated separately.
- Final model selection should consider global accuracy, fold stability, FSA stability, horizon stability, extreme-condition behavior, and computational cost together.

---

## 19. Supporting Evidence

Detailed numerical results and diagnostic visualizations are available in the LightGBM modeling outputs and notebooks, particularly:

- `LGBM_03_training.ipynb`
- `LGBM_04_hyperparameter_tuning.ipynb`
- `LGBM_05_evaluation.ipynb`
- `LGBM_06_interpretation.ipynb`
- `LGBM_04_tuning_details.csv`
- `LGBM_04_tuning_summary.csv`
- `LGBM_05_global_metrics.csv`
- `LGBM_05_fold_metrics.csv`
- `LGBM_05_fsa_metrics.csv`
- `LGBM_05_horizon_metrics.csv`
- `LGBM_05_monthly_fsa_metrics.csv`
- `LGBM_06_feature_importance.csv`

Recommended figures to retain as primary evidence are:

1. **MAE by Forecast Horizon**
2. **FSA × Month MAE heatmap**
3. **FSA × Fold MAE heatmap**
4. **M9W-2023 Actual vs Predicted Timeline**
5. **M9W Actual vs Predicted Monthly Level**
6. **Forecast Error by Temperature Band**
7. **Feature Importance**

Together, these figures summarize forecast-distance degradation, seasonal behavior, spatial stability, the M9W stress case, temperature-related difficulty, and predictor usage.
