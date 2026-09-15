# HistGradientBoostingClassifier — Peak-Risk Candidate Model Overview

## 1. Purpose

The **HistGradientBoostingClassifier (HGB)** was developed as a candidate model for the Peak-Risk classification task.

Its purpose is to estimate the probability that each of the next 24 hourly periods will be classified as a Peak:

`peak_h01 ... peak_h24`

The model follows the same Modeling Foundation rules used by the Logistic Regression baseline so that both models can later be compared under equivalent conditions.

The final **2025 holdout was not evaluated** during this stage.

---

## 2. Conceptual Modeling Process

The process can be understood as follows:

1. Historical electricity-demand information and contextual variables available at the forecast origin are prepared.
2. For each training fold, the Peak definition is calculated using the project's **97.5th-percentile rule by FSA + season**.
3. The threshold is calculated using training information only and is then applied unchanged to the corresponding validation period.
4. A separate HGB classifier is trained for each forecast horizon from `h+1` through `h+24`.
5. The model learns combinations of historical demand, time, FSA, and weather conditions that are associated with Peak events.
6. A moderate hyperparameter search is performed using development folds only.
7. The selected configuration is evaluated globally and by fold, FSA, month, and forecast horizon.
8. Additional diagnostic analysis is performed using the predicted Peak probabilities, including threshold sensitivity, Precision-Recall behavior, class imbalance, false negatives, and Peak-detection timelines.

HistGradientBoosting works by building a sequence of decision trees in which each new stage attempts to improve the errors left by the previous stages. This allows the model to capture nonlinear relationships and interactions between predictors.

---

## 3. Peak-Risk Definition

A Peak is not defined using a single fixed consumption value for the complete dataset.

The project uses:

- **97.5th percentile**
- calculated by **FSA + season**
- estimated using **training data only**

This approach allows zones with different demand scales to have context-specific Peak thresholds and prevents future information from influencing the target definition.

For the initial standardized comparison between classifiers, the probability threshold remains:

`0.50`

This converts predicted probabilities into:

- `Peak = 1`
- `No Peak = 0`

Alternative probability thresholds were analyzed diagnostically but were not adopted as part of the official model configuration.

---

## 4. Hyperparameter Selection

A moderate tuning process compared several HistGradientBoosting configurations using representative forecast horizons and the common validation folds.

The selected configuration was:

| Parameter | Selected value |
|---|---:|
| Learning rate | 0.08 |
| Maximum iterations | 160 |
| Maximum leaf nodes | 31 |
| Minimum samples per leaf | 40 |
| L2 regularization | 1.0 |

The selected configuration obtained the strongest mean PR-AUC among the tested alternatives.

No additional tuning was performed at this stage. Further optimization may be reconsidered only after all candidate Peak-Risk models have been compared.

---

## 5. Overall Validation Results

Across the complete development validation predictions, HGB obtained:

| Metric | Result |
|---|---:|
| Precision | **0.823** |
| Recall | **0.208** |
| F1 | **0.332** |
| PR-AUC | **0.580** |
| ROC-AUC | **0.909** |
| Balanced Accuracy | **0.602** |
| Actual Peak Rate | **7.48%** |
| Validation predictions | **2,519,424** |

### Technical Interpretation

The model has strong overall ranking/discrimination ability, particularly as shown by ROC-AUC, but its behavior at the fixed `0.50` classification threshold is conservative.

Precision is high, meaning that when the model predicts a Peak, the prediction is usually correct.

Recall is considerably lower, meaning that many actual Peak events are not classified as Peaks at the current threshold.

### Non-Technical Interpretation

The model is cautious when issuing a Peak alert.

When it raises an alert, it is often correct. However, it misses many real Peak events because it requires a relatively high predicted probability before declaring a Peak.

Therefore, the main weakness is not excessive false alarms. The main weakness is **missed Peaks**.

---

## 6. Validation Fold Performance

| Fold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 2023 | 0.903 | 0.141 | 0.244 | 0.625 | 0.901 |
| 2024 | 0.756 | 0.389 | 0.514 | 0.682 | 0.980 |

The model performs substantially better in the 2024 validation fold.

Peak prevalence also changes considerably:

- **2023:** approximately 10.97%
- **2024:** approximately 4.01%

This demonstrates that the Peak-Risk problem is temporally imbalanced and that model performance should not be evaluated only through one pooled global result.

---

## 7. Class Imbalance

Peak events represent a minority of observations.

The development folds contain:

| Fold | No Peak | Peak |
|---|---:|---:|
| 2023 | 89.03% | 10.97% |
| 2024 | 95.99% | 4.01% |

This imbalance explains why metrics such as Accuracy would provide an incomplete view of performance.

For this reason, the project emphasizes:

- PR-AUC
- Precision
- Recall
- F1

rather than relying primarily on overall Accuracy.

---

## 8. Classification Threshold Analysis

The official comparison threshold is `0.50`.

At this threshold:

- Precision = **0.823**
- Recall = **0.208**
- F1 = **0.332**
- predicted Peak rate = **1.89%**
- actual Peak rate = **7.48%**

The model therefore predicts considerably fewer Peaks than actually occur.

A diagnostic threshold sweep produced the following pattern:

| Threshold | Precision | Recall | F1 | Predicted Peak Rate |
|---:|---:|---:|---:|---:|
| 0.05 | 0.534 | 0.578 | **0.555** | 8.11% |
| 0.10 | 0.620 | 0.487 | 0.545 | 5.87% |
| 0.20 | 0.708 | 0.385 | 0.498 | 4.07% |
| 0.30 | 0.759 | 0.315 | 0.445 | 3.11% |
| 0.50 | **0.823** | 0.208 | 0.332 | 1.89% |

The highest F1 in the tested range occurred around a threshold of **0.05**.

### Interpretation

This does **not** mean that `0.05` should automatically replace `0.50`.

The analysis demonstrates that the model's probabilities contain useful information below the current decision threshold. Lowering the threshold would identify substantially more real Peaks, but it would also increase false alarms.

The `0.50` threshold is retained for the standardized model comparison. A different **operational threshold** may be selected later using validation data only.

PR-AUC itself does not change when the classification threshold changes because it evaluates the ranking of probabilities rather than a single binary decision point.

---

## 9. Confusion-Matrix Interpretation

At the `0.50` threshold, the complete validation predictions contain approximately:

| Result | Count |
|---|---:|
| True Negative | 2,322,478 |
| False Negative | 149,380 |
| True Positive | 39,137 |
| False Positive | 8,429 |

The dominant classification error is therefore **False Negative**.

### Non-Technical Interpretation

The model produces relatively few false alarms, but a large number of actual Peaks remain undetected.

For an operational Peak-Risk tool, the final decision threshold will therefore require a deliberate trade-off between:

- detecting more real Peaks; and
- avoiding excessive false alerts.

---

## 10. Performance Across FSAs

Peak prevalence differs substantially across the six FSAs.

A particularly important case is **M9W**.

### M9W — 2023

- Peak Rate: **51.40%**
- PR-AUC: **0.896**
- Precision: **0.991**
- Recall: **0.114**
- Missed Peak Rate: **88.62%**

### M9W — 2024

- Peak Rate: **10.57%**
- PR-AUC: **0.727**
- Precision: **0.781**
- Recall: **0.414**
- Missed Peak Rate: **58.64%**

M9W therefore behaves very differently from the remaining FSAs and from one year to the next.

### Interpretation

A high PR-AUC together with low Recall at the `0.50` threshold means that the probability ranking may still be informative even when relatively few events cross the classification threshold.

M9W remains an important stress test for all Peak-Risk models and should be examined separately during final model comparison.

---

## 11. Forecast-Horizon Behavior

Predictive performance generally decreases as the forecast horizon becomes longer.

The strongest results are usually observed for the closest horizons such as `h+1`, while later horizons show lower Recall and PR-AUC.

For example, in the 2024 validation fold:

- `h+1` Recall is approximately **0.594**
- `h+12` Recall is approximately **0.341**
- `h+24` Recall is approximately **0.331**

### Interpretation

Peak-Risk is easier to estimate for the near future than for events approaching 24 hours ahead.

This supports the project's decision to evaluate every forecast horizon separately rather than relying only on one global classification score.

---

## 12. Monthly Performance

The monthly analysis shows substantial variation in Peak prevalence and model performance.

For example, in 2024:

- January PR-AUC: approximately **0.872**
- May PR-AUC: approximately **0.820**
- June PR-AUC: approximately **0.766**
- September PR-AUC: approximately **0.726**

Other months are much more difficult:

- February PR-AUC: approximately **0.307**
- April PR-AUC: approximately **0.110**
- August PR-AUC: approximately **0.504**

October 2024 contains no actual Peak observations, and PR-AUC is therefore undefined for that month.

### Interpretation

Monthly metrics must always be interpreted together with Peak prevalence.

When very few Peaks occur, Recall, F1, and PR-AUC may become unstable. A month with no positive cases cannot provide a meaningful Precision-Recall evaluation.

The observed variability reinforces the need to evaluate temporal stability rather than selecting a model from one aggregate metric.

---

## 13. Predictor Importance

Permutation Importance identified the following leading predictors:

| Predictor | Importance |
|---|---:|
| Consumption lag 24h | 0.295 |
| Temperature | 0.221 |
| Month cyclic component | 0.128 |
| Target hour | 0.128 |
| Rolling mean 24h | 0.097 |
| Hour cyclic component | 0.077 |
| Reported premise count | 0.064 |
| FSA | 0.053 |
| Rolling standard deviation 24h | 0.043 |

### Interpretation

Historical electricity demand remains the strongest individual predictive signal, particularly consumption from the corresponding hour 24 hours earlier.

However, **temperature is the second most influential predictor** in this Peak-Risk model.

This is an important difference from the Random Forest demand-forecasting model, where historical demand dominated more strongly.

For Peak-Risk, the results suggest that extreme-demand conditions depend on a combination of:

- recent electricity-demand behavior;
- temperature;
- time of day;
- seasonal context;
- geographic area.

Humidity contributes less than temperature, while station pressure has a relatively small marginal contribution in the current model.

Permutation Importance measures predictor contribution within this fitted model and should not be interpreted as proof of causal relationships.

---

## 14. Visual Diagnostics

The evaluation notebook includes several visual diagnostics that complement the numerical metrics:

1. **Precision-Recall Curve**  
   Shows the trade-off between identifying more Peaks and maintaining accurate Peak alerts.

2. **ROC Curve**  
   Provides a complementary view of the model's ability to separate the two classes.

3. **Precision / Recall / F1 vs Threshold**  
   Demonstrates why the `0.50` threshold produces high Precision but relatively low Recall.

4. **Confusion Matrix**  
   Makes the high number of missed Peaks visible.

5. **Peak Rate by FSA**  
   Highlights the strong spatial variation, particularly M9W.

6. **False Negative Rate by FSA**  
   Shows where actual Peak events are most frequently missed.

7. **PR-AUC by Forecast Horizon**  
   Shows the decline in classification performance as the forecast horizon becomes longer.

8. **Monthly PR-AUC and Recall**  
   Identifies temporal periods where Peak detection is more difficult.

9. **Predicted Probability Distribution**  
   Shows how predicted risk differs between actual Peak and No-Peak observations.

10. **Peak-Detection Timeline**  
    Provides an intuitive view of correctly detected and missed Peak events over time.

These visualizations should be considered diagnostic evidence rather than independent model-selection criteria.

---

## 15. Comparison with the Logistic Regression Baseline

The Logistic Regression baseline achieved an overall PR-AUC of approximately **0.675**, while HGB achieved approximately **0.580**.

Therefore, HGB does **not** currently outperform the baseline on the project's principal global Peak-Risk model-selection metric.

However, performance differs across periods:

- HGB performs more strongly in 2024 than in 2023.
- Some FSA and horizon results are comparatively strong.
- HGB captures nonlinear relationships and shows substantial temperature importance.

### Interpretation

The candidate model should not be rejected simply because its pooled global metric is lower.

Its results demonstrate why final model selection must consider:

- global performance;
- fold stability;
- FSA stability;
- horizon stability;
- class-imbalance behavior;
- computational cost.

The remaining candidate classifiers should be evaluated before any final selection is made.

---

## 16. Main Strengths

The HGB candidate provides several positive characteristics:

- It models nonlinear relationships and interactions.
- It has strong ROC-AUC and meaningful PR-AUC under class imbalance.
- Performance improves substantially in the 2024 fold.
- It produces probabilities for all 24 Peak-Risk horizons.
- Temperature emerges as an important Peak-Risk predictor.
- It is computationally efficient relative to the scale of the evaluation.
- It provides interpretable diagnostic outputs through permutation importance.
- Threshold analysis shows that the model can achieve substantially higher Recall if the operational decision threshold is lowered.

---

## 17. Main Limitations and Observations

The current model also has important limitations:

1. **Global PR-AUC is lower than the Logistic Regression baseline.**
2. **Recall is low at the standardized 0.50 threshold.**
3. False Negatives represent the principal classification error.
4. Performance varies materially between 2023 and 2024.
5. Performance varies significantly across FSAs.
6. M9W presents an unusual Peak distribution and remains a major stress case.
7. Classification performance generally decreases with forecast horizon.
8. Monthly metrics can be unstable when very few Peaks occur.
9. The optimal operational classification threshold has not yet been selected.
10. Additional tuning or imbalance-aware training may improve the model but was intentionally deferred.

These limitations do not invalidate the model. They define the conditions under which subsequent Peak-Risk candidates should be compared.

---

## 18. Current Model Decision

**Accepted as a valid Peak-Risk candidate model.**

No additional hyperparameter tuning is performed at this stage.

The official comparison threshold remains **0.50** so that HGB can be compared under the same rules as the other classifiers.

The diagnostic analysis indicates that lower thresholds, particularly around `0.05–0.10`, substantially improve Recall and F1. This finding should be retained for the later operational-threshold selection stage rather than incorporated into the current standardized model comparison.

The final 2025 holdout remains protected and has **not** been used to evaluate or select this model.

---

## 19. Modeling Implications

The HGB experiment provides several useful conclusions for subsequent Peak-Risk modeling:

- Short-term historical demand is a strong predictor of Peak events.
- Temperature contributes materially to Peak-Risk classification.
- Class imbalance must remain a central consideration.
- PR-AUC should remain the primary model-selection metric.
- Recall must be monitored because a model with strong Precision may still miss many critical events.
- Probability-threshold selection is a separate operational decision from model ranking.
- M9W should remain a specific robustness test.
- Candidate models should be compared by fold, FSA, horizon, and temporal stability rather than only by one pooled score.
- If HGB remains competitive after all candidates are evaluated, future refinement could investigate class/sample weighting and operational threshold calibration.


