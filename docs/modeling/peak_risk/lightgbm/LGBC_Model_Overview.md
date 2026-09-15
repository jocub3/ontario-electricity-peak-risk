# LightGBM Classifier — Peak-Risk Candidate Model Overview

## 1. Purpose

The **LightGBM Classifier** was developed as a candidate model for the Peak-Risk classification task. Its objective is to estimate the probability that each forecast horizon from **h+1 through h+24** will correspond to a Peak hour.

The model follows the common Modeling Foundation used by all Peak-Risk candidates:

- Peak definition based on the **P97.5 threshold by FSA + season**;
- thresholds calculated from training information only;
- 24 direct forecast horizons;
- expanding-window temporal validation;
- PR-AUC as the primary candidate-selection metric;
- 0.50 as the common classification threshold for candidate comparison;
- final 2025 holdout protected.

---

## 2. Model Concept

LightGBM is a gradient-boosted decision-tree model. It builds trees sequentially, with each new tree attempting to improve errors left by the previous trees.

Conceptually:

> the model progressively combines many small decision trees to learn nonlinear relationships between historical demand, weather, calendar conditions, FSA context, and Peak occurrence.

This makes LightGBM appropriate for Peak-Risk because the relationships between these predictors are unlikely to be purely linear and may involve complex interactions.

---

## 3. Data and Predictor Strategy

The model uses leakage-safe information available at the forecast origin or deterministically known for the target timestamp.

Predictors include:

- historical electricity-demand lags;
- rolling demand statistics;
- hour, weekday, month, and season;
- cyclical calendar encodings;
- weather conditions available at forecast origin;
- reported premise count when available;
- FSA information.

Numeric standardization is not required because LightGBM is tree-based. Learned preprocessing remains within the training workflow so validation information does not influence model preparation.

---

## 4. Peak Target and Class Imbalance

Peak observations follow the project definition:

> **Peak = electricity consumption above the P97.5 threshold for the corresponding FSA and season.**

The threshold is recalculated inside each training window and then applied to validation observations.

Class imbalance is addressed using training-derived weighting so that the relatively uncommon Peak class receives appropriate attention without using validation information.

---

## 5. Hyperparameter Tuning

A moderate tuning process was performed using the development validation folds.

The selected configuration was:

| Parameter | Selected value |
|---|---:|
| Number of estimators | **300** |
| Number of leaves | **31** |
| Learning rate | **0.05** |
| Maximum depth | **Unlimited (-1)** |
| Minimum child samples | **20** |
| Subsample | **0.80** |
| Column sample by tree | **0.80** |
| L2 regularization | **0.0** |
| Imbalance multiplier | **1.0** |

The tested configurations produced very similar mean PR-AUC values. The selected configuration was the best according to the predefined selection rule, but the small differences indicate that extensive additional tuning is unlikely to be the main opportunity for improvement at this stage.

---

## 6. Overall Validation Results

Across the complete validation dataset:

| Metric | Result |
|---|---:|
| Precision | **0.5553** |
| Recall | **0.4704** |
| F1 | **0.5093** |
| Balanced Accuracy | **0.7200** |
| PR-AUC | **0.5324** |
| ROC-AUC | **0.9092** |
| Brier Score | **0.0533** |
| Actual Peak Rate | **7.48%** |
| Validation observations | **2,519,424** |

### Technical Interpretation

The model provides useful discrimination in a strongly imbalanced classification problem. PR-AUC of approximately **0.53** is substantially above the overall Peak prevalence of approximately 7.5%, while ROC-AUC is approximately **0.91**.

At threshold 0.50, the model provides a relatively high recall compared with a more conservative classifier, although this comes at the cost of additional false-positive alerts.

### Non-Technical Interpretation

LightGBM can identify many periods with elevated Peak risk. It is relatively willing to generate Peak alerts, which helps detect more real Peaks but also produces more false alarms.

---

## 7. Validation Fold Performance

The model behaves differently across the two validation periods.

### 2023

The model is comparatively conservative:

- Precision ≈ **0.717**
- Recall ≈ **0.326**
- F1 ≈ **0.448**
- PR-AUC ≈ **0.593**
- ROC-AUC ≈ **0.910**

### 2024

The behavior changes substantially:

- Precision ≈ **0.451**
- Recall ≈ **0.865**
- F1 ≈ **0.593**
- PR-AUC ≈ **0.666**
- ROC-AUC ≈ **0.979**

### Interpretation

In 2023 the model generates fewer positive classifications and misses many Peaks. In 2024 it detects most Peaks but generates more false-positive alerts.

This confirms that temporal distribution and calibration changes materially affect the behavior of a fixed probability threshold.

---

## 8. M9W-2023 Structural Stress Case

M9W-2023 again represents the most important structural stress case.

Its observed Peak Rate is approximately:

> **51.40%**

For M9W-2023, LightGBM produced:

| Metric | Result |
|---|---:|
| Precision | **0.9789** |
| Recall | **0.2712** |
| F1 | **0.4247** |
| PR-AUC | **0.9079** |
| ROC-AUC | **0.9061** |
| Brier Score | **0.3254** |

### Interpretation

The very high PR-AUC indicates that LightGBM still ranks high-risk M9W observations effectively.

However, recall at threshold 0.50 is only approximately 27%. Therefore, the model contains useful predictive information but its probability scale is poorly aligned with the major M9W-2023 regime change.

This is consistent with the structural issue previously identified during Target Definition and the other modeling experiments.

---

## 9. Impact of M9W-2023 on Global Results

The additional diagnostic compared the full validation result with performance after excluding M9W-2023 **for diagnostic purposes only**.

| Segment | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| All validation data | 0.5553 | 0.4704 | 0.5093 | 0.5324 | 0.9092 | 0.0533 |
| **Excluding M9W-2023** | **0.4579** | **0.7363** | **0.5646** | **0.6009** | **0.9643** | **0.0287** |
| M9W-2023 only | 0.9789 | 0.2712 | 0.4247 | 0.9079 | 0.9061 | 0.3254 |

### Interpretation

M9W-2023 materially affects pooled performance.

When it is excluded only as a sensitivity diagnostic:

- recall rises from approximately **0.47 to 0.74**;
- F1 rises from approximately **0.51 to 0.56**;
- PR-AUC and ROC-AUC improve;
- Brier Score improves from approximately **0.053 to 0.029**.

M9W-2023 must remain in the official evaluation because it represents a real temporal-generalization challenge rather than a record that should simply be removed.

---

## 10. Macro-Average Performance Across FSAs

Giving every FSA equal weight produces:

| Fold | Macro Precision | Macro Recall | Macro F1 | Macro PR-AUC | Macro ROC-AUC | Macro Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | **0.5622** | **0.5211** | **0.4833** | **0.6047** | **0.9482** | **0.7531** |
| 2024 | **0.4191** | **0.8539** | **0.5593** | **0.6376** | **0.9783** | **0.9045** |

### Interpretation

The macro results confirm that the temporal difference is not merely caused by one large FSA dominating the pooled observations.

The model becomes substantially more sensitive in 2024 while maintaining strong ranking performance across FSAs.

---

## 11. Threshold Analysis and Temporal Stability

The common project threshold remains:

> **0.50**

The additional fold-specific analysis found:

| Fold | F1-optimal diagnostic threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **2023** | **0.05** | 0.5077 | 0.6149 | **0.5562** |
| **2024** | **0.75** | 0.5737 | 0.7043 | **0.6323** |

At threshold 0.50:

- 2023: Precision ≈ **0.717**, Recall ≈ **0.326**, F1 ≈ **0.448**
- 2024: Precision ≈ **0.451**, Recall ≈ **0.865**, F1 ≈ **0.593**

### Interpretation

This is one of the most important findings from the LightGBM experiment.

The threshold that maximizes F1 changes from approximately **0.05 in 2023** to **0.75 in 2024**.

Therefore, there is no evidence that a single optimized probability threshold would remain stable across time.

The common 0.50 threshold should continue to be used for fair candidate comparison. Operational threshold optimization should occur only after final model selection and should explicitly consider temporal robustness and the relative cost of missed Peaks versus false alarms.

---

## 12. Performance Across the 24-Hour Horizon

LightGBM maintains useful Peak-Risk signal across all 24 forecast horizons.

In the 2024 validation fold, recall remains particularly high even at longer horizons. Precision generally decreases as the horizon increases.

Conceptually:

> longer-horizon forecasts continue to capture many actual Peaks, but they generate progressively more false-positive alerts.

This means horizon-specific behavior should remain an explicit component of the final classifier comparison.

---

## 13. Feature Importance

The strongest LightGBM predictors were:

1. **Temperature at forecast origin**
2. **Station pressure**
3. **Reported premise count**
4. **Demand lag 24h**
5. **Relative humidity**
6. **Rolling demand standard deviation 24h**
7. **Rolling demand mean 24h**
8. **Rolling demand mean 168h**
9. **Demand lag 168h**
10. **Target hour**

Other relevant variables include month, weekday, additional demand lags, cyclical time variables, and the M9W indicator.

### Interpretation

LightGBM uses a broad combination of:

- weather;
- recent demand history;
- demand variability;
- calendar context;
- FSA information.

Temperature is the strongest model-native feature, while pressure and humidity also appear prominently. This supports the predictive relevance of weather information in the Peak-Risk framework.

Feature importance does **not** demonstrate causality. If LightGBM becomes a finalist, SHAP-based interpretation could be considered later.

The importance of `reported_premise_count` also reinforces the need to confirm that this variable will be available or can be safely supplied at operational forecast time.

---

## 14. Probability Calibration

The Brier Score changes substantially depending on the validation context:

- all validation: **0.0533**
- excluding M9W-2023: **0.0287**
- M9W-2023: **0.3254**

Together with the threshold analysis, this shows that LightGBM's probability scale is sensitive to temporal regime changes.

Therefore:

> discrimination and probability calibration must be treated as separate model properties.

A model may rank Peak observations well while still requiring later probability calibration or operational threshold design.

---

## 15. Main Strengths

The LightGBM candidate demonstrates:

- useful PR-AUC in an imbalanced classification problem;
- strong ROC-AUC;
- very high recall during the 2024 validation period;
- predictive signal across all 24 horizons;
- effective nonlinear modeling;
- meaningful use of weather and historical-demand information;
- explicit class-imbalance handling;
- training-only Peak thresholds;
- leakage-safe temporal validation;
- standardized outputs suitable for direct model comparison;
- strong ranking performance even within M9W-2023.

---

## 16. Main Limitations

Important limitations are:

1. Global PR-AUC is lower than the strongest candidate results observed so far.
2. Probability calibration changes considerably across validation periods.
3. M9W-2023 materially affects pooled metrics.
4. The F1-optimal threshold is extremely unstable between 2023 and 2024.
5. High recall at longer horizons comes with reduced precision.
6. Performance varies across FSAs and time.
7. Model-native feature importance is not causal.
8. `reported_premise_count` is influential and its future operational availability must remain documented.
9. Further tuning may yield incremental improvement, but temporal calibration appears more important than additional model complexity.

---

## 17. Hyperparameter and Threshold Decision

### Hyperparameter Tuning

**No additional hyperparameter tuning is recommended at this stage.**

The moderate tuning search produced closely grouped candidate results. There is insufficient evidence that a larger search would materially change the model's standing before comparison with the other classifiers.

Further tuning should only be reconsidered if LightGBM becomes a finalist or near-finalist.

### Classification Threshold

**The official comparison threshold remains 0.50.**

The F1-optimal diagnostic threshold changes dramatically between validation folds:

- 2023: approximately **0.05**
- 2024: approximately **0.75**

Selecting one pooled optimized threshold now would therefore risk overfitting the development period.

Operational threshold selection should remain a separate post-model-selection stage.

---

## 18. Current Model Decision

**Status: Accepted as a valid Peak-Risk candidate model.**

The LightGBM Classifier is technically complete and ready for formal model comparison.

No additional retraining, hyperparameter tuning, or threshold optimization is required before proceeding.

The final 2025 holdout remains protected.

---

## 19. Modeling Implications

The LightGBM experiment contributes several conclusions to the overall Peak-Risk modeling process:

- Peak-Risk contains meaningful nonlinear predictive signal.
- Weather variables contribute substantial predictive information.
- Historical electricity demand remains important.
- PR-AUC should remain the primary candidate-selection metric.
- Recall alone should not determine the winning classifier.
- Calibration stability must be considered separately from ranking performance.
- M9W-2023 should remain a formal structural stress test.
- Threshold optimization cannot be treated as a simple pooled optimization problem.
- Performance must be examined by fold, FSA, and forecast horizon.
- Final model selection should consider discrimination, recall, precision, calibration, temporal stability, spatial stability, and horizon behavior together.

---

## 20. Supporting Evidence

Detailed outputs and visual diagnostics are available in the LightGBM branch, particularly:

- `LGBC_03_training.ipynb`
- `LGBC_04_hyperparameter_tuning.ipynb`
- `LGBC_05_evaluation.ipynb`
- `LGBC_06_interpretation.ipynb`
- `LGBC_04_tuning_details.csv`
- `LGBC_04_tuning_summary.csv`
- `LGBC_05_global_metrics.csv`
- `LGBC_05_fold_metrics.csv`
- `LGBC_05_fsa_metrics.csv`
- `LGBC_05_horizon_metrics.csv`
- `LGBC_05_class_balance.csv`
- `LGBC_05_threshold_analysis.csv`
- `LGBC_06_feature_importance.csv`

Recommended figures to retain as primary evidence are:

1. **Precision-Recall Curves**
2. **ROC Curves**
3. **Confusion Matrices**
4. **PR-AUC by Forecast Horizon**
5. **Peak Rate by FSA**
6. **False Negative Rate by FSA**
7. **Threshold Diagnostics by Validation Fold**
8. **Calibration Curves**
9. **Top Feature Importances**

Together, these provide the evidence required to understand discrimination, class imbalance, spatial variation, horizon behavior, calibration, threshold stability, and the predictor structure of the LightGBM candidate.
