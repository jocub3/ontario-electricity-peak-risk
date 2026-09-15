# Random Forest Classifier — Peak-Risk Candidate Model Overview

## 1. Purpose

The **Random Forest Classifier** was developed as a candidate model for the Peak-Risk classification task.

Its objective is to estimate the probability that each forecast horizon from **h+1 through h+24** will correspond to a Peak hour.

The model follows the common Modeling Foundation used by the other Peak-Risk candidates:

- Peak definition based on the **P97.5 threshold by FSA + season**;
- thresholds fitted using training data only;
- 24 direct forecast horizons;
- expanding-window validation;
- PR-AUC as the primary candidate-selection metric;
- 0.50 as the common comparison threshold;
- final 2025 holdout protected.

---

## 2. Model Concept

Random Forest is an ensemble classification model composed of many decision trees.

Each tree is trained on a different sample of the training data and evaluates a different subset of predictor variables. The final Peak probability is obtained by aggregating the outputs of the individual trees.

Conceptually:

> many different decision trees evaluate different combinations of demand history, time, weather, and FSA context, and their combined decision produces the final Peak-Risk probability.

This makes Random Forest suitable for Peak-Risk because the relationship between Peak events and their predictors is not expected to be purely linear.

---

## 3. Data and Predictor Strategy

The model uses leakage-safe information available at the forecast origin or deterministically known for the target timestamp.

The predictor set includes:

- historical demand lags;
- rolling demand statistics;
- target hour, weekday, month, and season;
- cyclical calendar encodings;
- weather conditions available at forecast origin;
- reported premise count when available;
- FSA information.

Numeric standardization is not required because Random Forest is tree-based.

Preprocessing that must be learned from the data, such as imputation and categorical encoding, remains inside the training pipeline.

---

## 4. Peak Target Construction

Peak labels are constructed using the common project definition:

> **Peak = electricity consumption above the P97.5 threshold for the corresponding FSA and season.**

The threshold is recalculated inside each training window.

Validation data is never used to estimate the Peak threshold.

This prevents future information from influencing target construction and preserves the anti-leakage design of the project.

---

## 5. Hyperparameter Tuning

A moderate tuning process was performed on representative horizons using the development folds.

The selected Random Forest configuration was:

| Parameter | Selected value |
|---|---:|
| Number of trees | **300** |
| Maximum depth | **20** |
| Minimum samples per leaf | **10** |
| Maximum features | **0.50** |
| Class weighting | **balanced** |

The selected configuration achieved the strongest mean PR-AUC among the evaluated candidate configurations.

The tuning stage therefore provided a reasonable configuration for model comparison without performing an exhaustive search.

No additional tuning is considered necessary at this stage.

---

## 6. Overall Validation Results

Across the complete validation dataset, the model produced:

| Metric | Result |
|---|---:|
| Precision | **0.6691** |
| Recall | **0.3553** |
| F1 | **0.4641** |
| Balanced Accuracy | **0.6706** |
| PR-AUC | **0.5791** |
| ROC-AUC | **0.9411** |
| Brier Score | **0.0446** |
| Actual Peak Rate | **7.48%** |
| Predicted Peak Rate | **3.97%** |

### Technical Interpretation

The model demonstrates strong ranking ability:

- ROC-AUC is approximately **0.94**;
- PR-AUC is approximately **0.58**, substantially above the global Peak prevalence of approximately 7.5%.

However, with the common comparison threshold of 0.50, the model is conservative.

It predicts fewer Peak observations than actually occur, which produces relatively high precision but comparatively low recall.

### Non-Technical Interpretation

When Random Forest predicts a Peak, it is often correct.

However, at the 0.50 threshold it misses a substantial number of actual Peak events.

This does not mean the model cannot identify high-risk observations. The threshold analysis shows that many of these Peaks receive elevated probabilities but remain below the common 0.50 cutoff.

---

## 7. Validation Fold Performance

Performance differs substantially between the two validation periods.

| Fold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 2023 | **0.7592** | **0.2333** | **0.3569** | **0.6284** | **0.9331** |
| 2024 | **0.6029** | **0.6881** | **0.6427** | **0.6727** | **0.9799** |

The actual Peak Rate also changes considerably:

- 2023: approximately **10.97%**
- 2024: approximately **4.01%**

### Interpretation

The model ranks Peak risk reasonably well in both years, as shown by PR-AUC and ROC-AUC.

However, the fixed 0.50 threshold behaves very differently across folds.

In 2023 the model is extremely conservative and misses many Peaks.

In 2024 the same threshold produces a substantially better balance between precision and recall.

This indicates a temporal calibration and distribution-shift issue rather than a complete loss of discriminative ability.

---

## 8. M9W-2023 Structural Stress Case

M9W-2023 is the most important special case in this model evaluation.

The actual Peak Rate for M9W during 2023 is approximately:

> **51.40%**

This is dramatically higher than the other FSAs during the same fold.

Representative 2023 Peak Rates were approximately:

| FSA | Peak Rate |
|---|---:|
| L4T | 1.90% |
| M5R | 1.97% |
| M5S | 1.58% |
| M6G | 5.57% |
| M9R | 3.36% |
| **M9W** | **51.40%** |

For M9W-2023, Random Forest achieved:

| Metric | Result |
|---|---:|
| Precision | **0.9880** |
| Recall | **0.1725** |
| F1 | **0.2937** |
| PR-AUC | **0.9277** |
| ROC-AUC | **0.9260** |
| Brier Score | **0.3022** |

### Interpretation

The very high PR-AUC indicates that the model ranks the M9W Peak observations extremely well.

However, recall at threshold 0.50 is only approximately 17%.

Therefore, the model can distinguish higher-risk M9W observations, but its probability scale is not aligned with the large regime change observed in 2023.

This is consistent with the structural change previously identified in M9W during the Target Definition and forecasting analyses.

---

## 9. M9W-2023 Impact on Global Metrics

A dedicated diagnostic compared the full validation result with the result obtained after excluding M9W-2023.

| Segment | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| All validation data | 0.6691 | 0.3553 | 0.4641 | 0.5791 | 0.9411 | 0.0446 |
| **Excluding M9W-2023** | **0.5952** | **0.5993** | **0.5973** | **0.6207** | **0.9724** | **0.0213** |
| M9W-2023 only | 0.9880 | 0.1725 | 0.2937 | 0.9277 | 0.9260 | 0.3022 |

### Interpretation

The global recall and F1 are strongly influenced by M9W-2023.

When that structural stress case is excluded:

- recall increases from approximately **0.36 to 0.60**;
- F1 increases from approximately **0.46 to 0.60**;
- Brier Score improves substantially.

This confirms that the global result should not be interpreted as uniform weakness across all FSAs.

At the same time, M9W-2023 should not be removed from the official evaluation because it represents a real temporal-generalization challenge.

---

## 10. Macro-Average Performance by FSA

Macro-averaging gives each FSA equal weight and reduces the influence of large or unusual groups.

| Fold | Macro Precision | Macro Recall | Macro F1 | Macro PR-AUC | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 2023 | 0.6385 | 0.4437 | 0.4726 | 0.6169 | 0.9593 |
| 2024 | 0.5752 | 0.6830 | 0.6218 | 0.6477 | 0.9800 |

### Interpretation

The macro results confirm the general fold-level conclusion.

Performance is considerably more balanced in 2024, while 2023 remains more difficult.

However, the model retains good ranking ability across FSAs in both validation periods.

---

## 11. Threshold Analysis

The common project threshold remains **0.50** for candidate-model comparison.

However, the diagnostic threshold analysis shows that 0.50 is not the threshold that maximizes F1.

For the complete validation set:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.10 | 0.404 | 0.827 | 0.543 |
| 0.15 | 0.459 | 0.739 | 0.566 |
| **0.20** | **0.498** | **0.661** | **0.568** |
| 0.25 | 0.530 | 0.592 | 0.559 |
| 0.30 | 0.562 | 0.538 | 0.550 |
| **0.50** | **0.669** | **0.355** | **0.464** |

A lower threshold therefore improves recall substantially.

### Threshold Stability by Fold

The threshold that maximizes F1 differs strongly by validation period:

| Fold | Best F1 Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **2023** | **0.10** | 0.5327 | 0.7712 | **0.6301** |
| **2024** | **0.45** | 0.5718 | 0.7472 | **0.6478** |

### Interpretation

This is an important result.

There is no single F1-optimal probability threshold that is stable across both validation folds.

The large difference between approximately 0.10 in 2023 and 0.45 in 2024 indicates substantial temporal calibration shift.

For that reason, the threshold should **not** be optimized now using all development predictions and then treated as universally valid.

The common 0.50 threshold should continue to be used for candidate-model comparison.

Operational threshold selection should be treated as a separate later stage and should explicitly consider temporal stability and business costs of missed Peaks versus false alarms.

---

## 12. Performance by Forecast Horizon

Random Forest retains useful predictive signal across the complete 24-hour horizon.

Performance does not collapse as the forecast horizon increases.

For example, in the 2024 validation fold, the model continues to produce strong ROC-AUC and useful PR-AUC even toward h+24.

At h+24, representative results are approximately:

- Precision: **0.55**
- Recall: **0.66**
- F1: **0.60**
- PR-AUC: **0.61**
- ROC-AUC: **0.98**

### Interpretation

Peak-Risk classification remains feasible across the full next-24-hour window.

This is an important distinction from consumption forecasting, where absolute prediction error generally grows more strongly with horizon.

---

## 13. Monthly Performance

The monthly evaluation reveals considerable variability.

Some periods produce strong PR-AUC but weak recall at threshold 0.50.

Other months contain very few or no Peak observations, making metrics such as precision and recall unstable or undefined in practical terms.

For example, some 2024 months have very low Peak prevalence, and October contains no observed Peaks in the evaluated aggregation.

### Interpretation

Monthly Peak-Risk metrics must always be interpreted together with the number and prevalence of actual Peaks.

A month with zero or very few Peaks should not be used alone to judge classifier quality.

The variability also reinforces the importance of PR-AUC and temporal validation rather than relying only on accuracy or a single monthly score.

---

## 14. Feature Importance

The strongest Random Forest predictors were:

| Feature | Importance |
|---|---:|
| Target hour sine | **0.2139** |
| Temperature at forecast origin | **0.1668** |
| Target hour | **0.1288** |
| Demand lag 24h | **0.1054** |
| Target hour cosine | 0.0337 |
| Rolling mean 24h | 0.0323 |
| Target month cosine | 0.0321 |
| Target month sine | 0.0319 |
| Rolling standard deviation 24h | 0.0290 |
| Demand lag 48h | 0.0238 |
| Reported premise count | 0.0237 |
| Station pressure | 0.0217 |
| Relative humidity | 0.0202 |
| Demand lag 168h | 0.0197 |

### Interpretation

The most important information comes from a combination of:

- **time of day**;
- **temperature**;
- **recent historical electricity demand**;
- **recent demand variability**;
- **seasonal/calendar information**.

Temperature is the second most important feature under the current Random Forest importance measure.

This supports the inclusion of weather information in the Peak-Risk framework.

However, tree feature importance does not imply causality and should not be interpreted as a percentage causal contribution.

If Random Forest becomes a finalist, SHAP-based interpretation could be considered later.

---

## 15. Probability Calibration

The calibration curves and Brier Score show that probability quality differs considerably between validation periods and contexts.

The global Brier Score is approximately **0.0446**, but the M9W-2023 Brier Score is approximately **0.3022**, while the score outside that structural case is approximately **0.0213**.

### Interpretation

Random Forest has strong discrimination, but probability calibration is sensitive to temporal regime changes.

This explains why threshold 0.50 behaves very differently between 2023 and 2024.

Therefore, discrimination and calibration should be treated as separate properties during final model comparison.

---

## 16. Main Strengths

The Random Forest Peak-Risk candidate demonstrates several strengths:

- strong global ROC-AUC;
- useful PR-AUC relative to the low Peak prevalence;
- good performance in the 2024 validation fold;
- useful predictive signal across all 24 forecast horizons;
- conceptually coherent feature importance;
- effective use of nonlinear relationships and interactions;
- explicit class-imbalance handling;
- training-only Peak thresholds;
- leakage-safe preprocessing;
- stable implementation with standardized outputs;
- strong ranking performance even in the difficult M9W-2023 case.

---

## 17. Main Limitations

The evaluation also identifies important limitations:

1. Recall is low at the common 0.50 threshold when results are pooled globally.
2. The model is strongly affected by the M9W-2023 structural regime change.
3. Probability calibration changes substantially between 2023 and 2024.
4. The F1-optimal threshold is not temporally stable.
5. Monthly performance varies considerably with Peak prevalence.
6. The global metrics can hide important FSA-specific differences.
7. Model-native feature importance is not causal.
8. Further tuning may provide incremental gains, but the dominant issue currently appears to be calibration and threshold stability rather than model capacity.

---

## 18. Hyperparameter and Threshold Decision

### Hyperparameter Tuning

**No additional hyperparameter tuning is recommended at this stage.**

The moderate search already selected a reasonable configuration, and the model demonstrates strong discrimination.

Additional tree-depth or estimator tuning should only be reconsidered if Random Forest becomes a finalist or near-finalist after comparison with HistGradientBoosting, XGBoost, LightGBM, and Logistic Regression.

### Classification Threshold

**The official comparison threshold remains 0.50.**

Although lower thresholds produce substantially better recall and F1 in some validation periods, the threshold that maximizes F1 is not stable across time.

Operational threshold selection must therefore be treated as a separate post-model-selection decision.

---

## 19. Current Model Decision

**Status: Accepted as a valid Peak-Risk candidate model.**

The Random Forest Classifier is technically complete and ready for later model comparison.

No additional analysis or retraining is required before continuing with the next Peak-Risk candidates.

The final 2025 holdout remains protected and has not been used for candidate selection or threshold optimization.

---

## 20. Modeling Implications

The Random Forest experiment provides several important conclusions for the remaining Peak-Risk workflow:

- Peak-Risk is predictable using nonlinear combinations of time, weather, and demand history.
- Time of day and temperature are particularly important predictive signals.
- PR-AUC should remain the principal comparison metric because Peak prevalence is low and variable.
- Recall at a fixed threshold can be misleading when probability calibration shifts over time.
- M9W-2023 should remain a formal structural stress test in the final comparison.
- Candidate models should be compared both on discrimination and calibration.
- Threshold optimization should occur only after model selection.
- A final operational threshold may need to prioritize recall if missing Peak events is considered more costly than false alarms.
- Model comparison should consider global, fold, FSA, horizon, and calibration stability rather than a single score.

---

## 21. Supporting Evidence

Detailed results and diagnostic visualizations are available in the Random Forest branch outputs and notebooks, particularly:

- `RFC_03_training.ipynb`
- `RFC_04_hyperparameter_tuning.ipynb`
- `RFC_05_evaluation.ipynb`
- `RFC_06_interpretation.ipynb`
- `RFC_04_tuning_details.csv`
- `RFC_04_tuning_summary.csv`
- `RFC_05_global_metrics.csv`
- `RFC_05_fold_metrics.csv`
- `RFC_05_fsa_metrics.csv`
- `RFC_05_horizon_metrics.csv`
- `RFC_05_class_balance.csv`
- `RFC_05_threshold_analysis.csv`
- `RFC_06_feature_importance.csv`

Recommended figures to retain as primary evidence are:

1. **Precision-Recall Curves**
2. **Confusion Matrices**
3. **PR-AUC by Forecast Horizon**
4. **Peak Rate by FSA**
5. **False Negative Rate by FSA**
6. **Threshold Diagnostics by Fold**
7. **Calibration Curves**
8. **Top Feature Importances**

Together, these figures summarize discrimination, threshold behavior, class imbalance, temporal calibration, spatial differences, and the main predictor signals.
