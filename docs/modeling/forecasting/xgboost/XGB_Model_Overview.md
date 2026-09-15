# XGBoost Regressor — Forecasting Candidate Model Overview

## 1. Purpose

The **XGBoost Regressor** was developed as a candidate model for the 24-hour electricity-demand forecasting task.

Its objective is to estimate electricity consumption independently for each forecast horizon from **h+1 through h+24**, while following the same Modeling Foundation rules used by the other forecasting models.

The model is evaluated using the common expanding-window validation framework, with results reported globally and by validation fold, FSA, and forecast horizon.

The final **2025 holdout was not evaluated** during this stage.

---

## 2. Conceptual Modeling Process

The XGBoost forecasting process can be understood as follows:

1. Historical electricity-demand information and contextual variables available at the forecast origin are prepared.
2. The model uses recent demand history, calendar context, FSA information, and available weather information to learn relationships associated with future electricity consumption.
3. A separate regression model is fitted for each forecast horizon from `h+1` through `h+24`.
4. A moderate hyperparameter search is performed using the development folds only.
5. The selected configuration is then evaluated across all 24 horizons.
6. Performance is reviewed globally and by fold, FSA, horizon, month, and selected stress cases.
7. Saved validation predictions are used for additional diagnostics without retraining the model.

XGBoost is a gradient-boosted decision-tree model. Conceptually, it builds trees sequentially, with later trees attempting to improve the errors left by earlier trees. This allows it to capture nonlinear relationships and interactions between electricity demand, time, weather, and geographic context.

---

## 3. Predictor Information

The model uses predictors that are either available at the forecast origin, derived from historical observations, or deterministically known for the target time.

The strongest model feature importances were:

| Feature | Importance |
|---|---:|
| Consumption lag 24h | **65.39%** |
| Consumption lag 48h | **18.66%** |
| Consumption lag 168h | **5.37%** |
| Temperature at forecast origin | 0.79% |
| Rolling mean 24h | 0.69% |
| FSA L4T indicator | 0.62% |
| Rolling standard deviation 24h | 0.52% |
| Target hour | 0.51% |
| Target hour cosine | 0.49% |
| Weekend indicator | 0.44% |

The three historical demand lags represent approximately **89.4%** of the total feature importance.

### Interpretation

The dominant forecasting signal is the recent historical behavior of electricity demand, particularly demand observed at the corresponding hour one day earlier.

This is consistent with strong daily and weekly persistence in electricity consumption.

Temperature is the most important weather variable in the current XGBoost model, but its marginal contribution is considerably smaller than the historical demand lags.

This does not demonstrate that weather is unimportant. The model currently uses weather information available at the forecast origin rather than future observed weather, which intentionally prevents data leakage.

---

## 4. Hyperparameter Selection

A moderate tuning process compared four XGBoost configurations using representative horizons from the shared development folds.

The selected configuration was:

| Parameter | Selected value |
|---|---:|
| Number of estimators | 400 |
| Maximum tree depth | 6 |
| Learning rate | 0.05 |
| Subsample | 0.80 |
| Column sample by tree | 0.80 |
| Minimum child weight | 5 |
| L2 regularization (`reg_lambda`) | 1.0 |

The tuning summaries were:

| Configuration | Mean MAE | Mean RMSE | Mean MAPE |
|---:|---:|---:|---:|
| **2 — selected** | **572.26** | 993.38 | **5.67%** |
| 3 | 580.70 | 1005.68 | 5.74% |
| 4 | 583.54 | **968.74** | 5.95% |
| 1 | 586.41 | 967.62 | 6.00% |

Configuration 2 was selected because it achieved the lowest mean MAE, the primary ranking criterion for this tuning stage.

The detailed tuning results also show that the same configuration is not necessarily optimal at every forecast horizon. This is retained as a possible future refinement opportunity, but no additional tuning is performed at this stage.

---

## 5. Overall Validation Results

Across all development validation predictions, XGBoost obtained:

| Metric | Result |
|---|---:|
| MAE | **606.27 kWh** |
| RMSE | **1,106.77 kWh** |
| MAPE | **5.93%** |
| WAPE | **6.66%** |
| Bias | **-251.05 kWh** |
| Validation predictions | **2,519,424** |

### Technical Interpretation

The model produces an average absolute error of approximately 606 kWh across the complete validation set.

MAPE is approximately 5.93%, indicating that the average absolute percentage error is close to 6%.

The negative official Bias indicates an overall tendency to **underpredict** demand under the project convention:

`Bias = Predicted - Actual`

### Non-Technical Interpretation

The model usually follows electricity demand reasonably closely, but it does not perform equally well in every year, FSA, or season.

A relatively small number of difficult periods, especially the M9W area during 2023, have a substantial influence on the global error.

---

## 6. Validation Fold Performance

| Fold | MAE (kWh) | RMSE (kWh) | MAPE | WAPE | Bias (kWh) |
|---|---:|---:|---:|---:|---:|
| 2023 | **741.73** | 1,351.14 | 6.86% | 8.22% | -447.92 |
| 2024 | **471.18** | 791.17 | 5.00% | 5.14% | -54.71 |

The model performs substantially better in 2024.

MAE falls from approximately **742 kWh to 471 kWh**, while the absolute Bias also becomes much smaller.

### Interpretation

This difference indicates that XGBoost is sensitive to changes in the underlying demand regime.

The improvement in 2024 is consistent with the expanding-window validation design, because the later model has access to additional historical information.

However, the fold comparison also reveals that the 2023 error is strongly affected by one FSA in particular: **M9W**.

---

## 7. Performance Across FSAs

### Fold 2023

| FSA | MAE (kWh) | MAPE | Bias (kWh) |
|---|---:|---:|---:|
| L4T | 653.94 | 5.41% | -173.13 |
| M5R | 410.58 | 5.36% | 62.49 |
| M5S | 191.67 | 4.72% | 5.49 |
| M6G | 674.64 | 5.44% | -446.10 |
| M9R | 356.26 | 5.93% | 1.51 |
| **M9W** | **2,163.29** | **14.32%** | **-2,137.79** |

### Fold 2024

| FSA | MAE (kWh) | MAPE | Bias (kWh) |
|---|---:|---:|---:|
| L4T | 624.01 | 4.97% | -147.90 |
| M5R | 387.06 | 5.09% | -1.24 |
| M5S | 194.69 | 4.87% | 36.69 |
| M6G | 561.08 | 4.67% | -97.11 |
| M9R | 355.55 | 5.78% | -45.33 |
| **M9W** | **704.66** | **4.60%** | **-73.36** |

Five FSAs show relatively consistent percentage errors, while **M9W-2023 is clearly different from the rest of the validation results**.

---

## 8. M9W-2023 Stress Case

A dedicated diagnostic analysis was performed because M9W had a much larger 2023 error than the remaining FSAs.

For M9W-2023:

| Metric | Result |
|---|---:|
| Observations | 209,664 |
| Mean actual demand | 14,259.20 kWh |
| Mean predicted demand | 12,121.41 kWh |
| MAE | **2,163.29 kWh** |
| MAPE | **14.32%** |
| Mean diagnostic residual | **+2,137.79 kWh** |
| Median diagnostic residual | 1,705.19 kWh |
| P95 absolute error | 5,750.85 kWh |
| Maximum absolute error | 16,277.43 kWh |

The additional diagnostic analysis defined:

`Residual = Actual - Predicted`

Therefore, the positive diagnostic residual indicates **underprediction**.

This is consistent with the official Bias of `-2,137.79 kWh`, because the project Bias convention uses the opposite sign:

`Bias = Predicted - Actual`

Both metrics describe the same behavior: **XGBoost substantially underpredicts M9W during 2023**.

### M9W versus the other FSAs in 2023

| Group | MAE | MAPE | Mean diagnostic residual |
|---|---:|---:|---:|
| **M9W** | **2,163.29 kWh** | **14.32%** | **+2,137.79 kWh** |
| Other five FSAs | **457.42 kWh** | **5.37%** | **+109.95 kWh** |

The M9W error is therefore almost five times the average MAE of the remaining FSAs.

### Interpretation

The poor 2023 global result is not a uniform failure across the complete dataset.

A large part of the deterioration is concentrated in M9W, where the model systematically predicts a lower level of demand than was actually observed.

This is treated as a genuine temporal-generalization stress case rather than an implementation error.

---

## 9. M9W Monthly Behavior

The monthly M9W analysis confirms that underprediction is present throughout 2023 and becomes particularly severe during summer.

| Month | Actual Mean | Predicted Mean | Difference | Difference % |
|---|---:|---:|---:|---:|
| January | 14,738.86 | 13,430.47 | 1,308.39 | 8.88% |
| February | 14,916.54 | 13,241.45 | 1,675.09 | 11.23% |
| March | 14,345.51 | 12,719.01 | 1,626.50 | 11.34% |
| April | 13,025.59 | 11,435.04 | 1,590.55 | 12.21% |
| May | 12,669.60 | 10,754.32 | 1,915.28 | 15.12% |
| June | 14,427.74 | 11,593.09 | 2,834.65 | 19.65% |
| **July** | **16,919.12** | **13,010.55** | **3,908.57** | **23.10%** |
| August | 14,915.90 | 11,923.28 | 2,992.63 | 20.06% |
| September | 13,939.56 | 10,909.35 | 3,030.21 | approximately 21.7% |
| October | 12,769.97 | 11,023.89 | 1,746.08 | approximately 13.7% |
| November | 13,886.06 | 12,265.83 | 1,620.23 | approximately 11.7% |
| December | 14,573.02 | 13,220.89 | 1,352.13 | approximately 9.3% |

### Interpretation

The model does not encounter a single isolated failure point.

Instead, M9W starts 2023 at a demand level that is already underestimated, and the difference becomes much larger from approximately **June through September**, with July representing the largest observed gap.

The Actual-vs-Predicted timeline confirms that the model follows part of the general shape of demand but fails to reproduce the higher demand level and several extreme periods.

---

## 10. Monthly Error Patterns

The full validation analysis also reveals a broader seasonal pattern beyond M9W.

### Fold 2023 — selected monthly MAE

| Month | MAE |
|---|---:|
| January | 484.86 kWh |
| May | 577.69 kWh |
| June | 947.68 kWh |
| **July** | **1,427.43 kWh** |
| August | 1,066.39 kWh |
| September | 1,043.51 kWh |
| December | 469.29 kWh |

### Fold 2024 — selected monthly pattern

The 2024 validation results again show clearly higher errors during the summer period, particularly June through August, even though the M9W structural mismatch is much smaller.

### Interpretation

Two effects are visible simultaneously:

1. **A general increase in forecasting difficulty during summer**, affecting several FSAs.
2. **An exceptional M9W-2023 level mismatch**, which greatly amplifies the 2023 global error.

This distinction is important. The 2023 deterioration should not be attributed entirely to M9W, because summer forecasting is also more difficult in the broader dataset.

---

## 11. Forecast-Horizon Behavior

Forecast accuracy generally decreases as the prediction horizon becomes longer.

For example, in the 2024 fold:

| Horizon | MAE | MAPE |
|---|---:|---:|
| h+1 | approximately **302 kWh** | **3.40%** |
| h+6 | approximately 407 kWh | 4.33% |
| h+12 | approximately **492 kWh** | **5.16%** |
| h+18 | approximately 536 kWh | 5.64% |
| h+24 | approximately **559 kWh** | **5.91%** |

The FSA × Horizon heatmaps confirm that this degradation is not restricted to one geographic area.

M9W-2023 remains difficult across multiple horizons, indicating that its problem is not simply a long-horizon forecasting issue.

### Interpretation

Predicting electricity demand one hour ahead is substantially easier than predicting demand 24 hours ahead.

This is an expected and realistic characteristic of multi-horizon forecasting and confirms the importance of reporting model performance separately for all 24 forecast horizons.

---

## 12. Comparison with Existing Forecasting Models

The current global results provide useful context:

| Model | MAE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|
| Seasonal Naive baseline | 612.05 | **994.29** | 6.59% | 6.73% |
| Random Forest Regressor | **542.26** | **926.18** | **5.55%** | **5.96%** |
| XGBoost Regressor | 606.27 | 1,106.77 | 5.93% | 6.66% |

Relative to Seasonal Naive, XGBoost:

- reduces MAE by approximately **0.9%**;
- reduces MAPE by approximately **10.0%**;
- reduces WAPE by approximately **0.9%**;
- has a higher RMSE, indicating larger extreme errors.

Random Forest currently performs better than XGBoost on the global forecasting metrics.

### Interpretation

XGBoost does add value over the simple baseline in percentage-error terms, but its global improvement is not consistently strong across every metric.

The large M9W-2023 errors have a particularly strong influence on RMSE and the pooled global result.

This model should therefore remain a valid candidate, but it should not currently be considered the leading forecasting model.

---

## 13. Residual and Diagnostic Analysis

The evaluation notebook includes several complementary diagnostics:

- residual distribution;
- Actual vs Predicted scatter plot;
- validation timeline;
- monthly MAE;
- monthly mean residual;
- FSA × Month MAE heatmaps;
- FSA × Fold heatmap;
- FSA × Horizon heatmaps;
- M9W monthly MAE;
- M9W residual timeline;
- M9W Actual vs Predicted timeline;
- M9W monthly actual-versus-predicted level comparison.

These diagnostics show that model error is not randomly distributed across all contexts.

Instead, forecasting difficulty is concentrated in identifiable situations, particularly:

- later forecast horizons;
- summer months;
- M9W during 2023.

---

## 14. Main Strengths

The XGBoost candidate demonstrates several positive characteristics:

- It successfully models all 24 forecast horizons.
- It follows the common Modeling Foundation and anti-leakage rules.
- Its behavior across most FSAs is reasonably consistent.
- It performs substantially better in the 2024 validation fold.
- Historical demand lags produce a conceptually coherent predictor structure.
- It captures nonlinear relationships and interactions without requiring numeric standardization.
- The model's deterioration with forecast horizon is realistic and interpretable.
- The standardized outputs allow direct comparison with Seasonal Naive, Random Forest, SARIMAX, and LightGBM.

---

## 15. Main Limitations and Observations

The evaluation identifies several important limitations:

1. **The global MAE improvement over Seasonal Naive is small.**
2. **Global RMSE is worse than the Seasonal Naive baseline**, indicating sensitivity to large forecasting errors.
3. Random Forest currently outperforms XGBoost on the main global forecasting metrics.
4. The model substantially underpredicts M9W during 2023.
5. M9W-2023 has much larger errors than the remaining FSAs.
6. Summer months are consistently more difficult in both validation years.
7. Error increases as the forecast horizon becomes longer.
8. The model depends strongly on historical consumption lags.
9. Weather variables provide relatively modest marginal importance under the current leakage-safe design.
10. Different forecast horizons may prefer different hyperparameter configurations.

These observations do not indicate that the model is incorrectly implemented. They define the conditions under which the model should be compared with the remaining forecasting candidates.

---

## 16. Current Model Decision

**Status: Accepted as a valid Forecasting candidate model.**

No additional hyperparameter tuning is performed at this stage.

The current XGBoost configuration is sufficiently developed for model comparison. Additional tuning should be reconsidered only if XGBoost becomes the best or a near-best candidate after all forecasting models have been evaluated.

Potential future refinements could include:

- more focused hyperparameter tuning;
- different parameter configurations for short-, medium-, and long-horizon forecasts;
- additional investigation of structural regime changes;
- use of genuine future weather forecasts when operationally available;
- SHAP-based interpretation if deeper explanation becomes necessary.

The final **2025 holdout remains protected** and has not been used for candidate selection.

---

## 17. Modeling Implications

The XGBoost experiment provides several useful conclusions for the remaining forecasting models:

- Historical demand from the previous day remains the dominant short-term predictive signal.
- Forecasting performance should not be judged only from a pooled global metric.
- M9W should remain a specific temporal-generalization stress test.
- Summer forecasting error should be an explicit comparison criterion across candidate models.
- Long-horizon performance should be compared separately from short-horizon performance.
- Candidate models that reduce extreme errors may improve RMSE even if average MAE is similar.
- A future model that performs well on M9W-2023 without sacrificing performance elsewhere would demonstrate stronger temporal robustness.
- Model selection should consider accuracy, temporal stability, spatial stability, horizon stability, and computational cost together.

---

## 18. Supporting Evidence

Detailed numerical results and diagnostic visualizations are available in the XGBoost modeling outputs and notebooks, particularly:

- `XGB_03_training.ipynb`
- `XGB_04_hyperparameter_tuning.ipynb`
- `XGB_05_evaluation.ipynb`
- `XGB_06_interpretation.ipynb`
- `XGB_04_tuning_details.csv`
- `XGB_04_tuning_summary.csv`
- `XGB_05_global_metrics.csv`
- `XGB_05_fold_metrics.csv`
- `XGB_05_fsa_metrics.csv`
- `XGB_05_horizon_metrics.csv`
- `XGB_05_monthly_fsa_metrics.csv`
- `XGB_06_feature_importance.csv`

Recommended figures to retain as primary evidence are:

1. **MAE by Forecast Horizon**
2. **Validation Actual vs Predicted Timeline**
3. **FSA × Month MAE heatmap**
4. **M9W-2023 Actual vs Predicted Timeline**
5. **M9W Actual vs Predicted Monthly Level**
6. **Feature Importance**

Together, these figures show overall forecasting behavior, horizon degradation, seasonal error patterns, the M9W structural stress case, and the variables used most strongly by the model.
