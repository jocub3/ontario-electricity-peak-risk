# XGBoost Classifier — Peak-Risk Candidate Model Overview

## 1. Purpose

The **XGBoost Classifier** was developed as a candidate model for the Peak-Risk classification task.

Its objective is to estimate the probability that each forecast horizon from **h+1 through h+24** will correspond to a Peak hour.

The model follows the common Modeling Foundation established for the project:

- Peak definition based on the **P97.5 threshold by FSA + season**;
- Peak thresholds fitted using training data only;
- 24 direct forecast horizons;
- expanding-window temporal validation;
- PR-AUC as the primary candidate-selection metric;
- 0.50 as the common probability threshold for fair classifier comparison;
- final 2025 holdout protected.

---

## 2. Model Concept

XGBoost is a gradient-boosted decision-tree classifier.

Unlike Random Forest, where many trees are trained relatively independently, XGBoost builds trees sequentially. Each new tree focuses on correcting errors that remain from the previous trees.

Conceptually:

> the model repeatedly identifies classification mistakes and adds new trees that improve the separation between Peak and No-Peak observations.

This makes XGBoost suitable for Peak-Risk because the relationship between electricity-demand history, weather, time, season, and FSA is likely to be nonlinear and interactive.

---

## 3. Data and Predictor Strategy

The model uses only information that is available at the forecast origin or deterministically known for the target timestamp.

The predictor set includes:

- historical demand lags;
- recent rolling demand statistics;
- target hour, weekday, month, and season;
- cyclical time encodings;
- weather conditions available at forecast origin;
- reported premise count when available;
- FSA indicators.

Numeric standardization is not required because XGBoost is tree-based.

Learned preprocessing operations remain inside the training pipeline so that validation information does not influence model preparation.

---

## 4. Peak Target Construction

The project Peak definition is:

> **Peak = electricity consumption above the P97.5 threshold for the corresponding FSA and season.**

The threshold is recalculated inside each training window and then applied unchanged to the corresponding validation observations.

This design prevents future information from entering the Peak definition and preserves the project's anti-leakage policy.

---

## 5. Class Imbalance Strategy

Peak observations are relatively uncommon in most FSA-period combinations.

XGBoost addresses this imbalance using a training-derived positive-class weight.

The weighting is calculated from the number of No-Peak and Peak observations available in the corresponding training data.

This encourages the model to pay greater attention to the minority Peak class without using validation information.

---

## 6. Hyperparameter Tuning

A moderate hyperparameter search was performed using representative horizons and the development validation folds.

The selected configuration was:

| Parameter | Selected value |
|---|---:|
| Number of estimators | **400** |
| Maximum depth | **6** |
| Learning rate | **0.05** |
| Subsample | **0.80** |
| Column sample by tree | **0.80** |
| Minimum child weight | **5** |
| L2 regularization | **1.0** |
| Imbalance multiplier | **1.0** |

The selected configuration achieved the highest mean PR-AUC among the tested configurations, although the two strongest alternatives were very close.

Therefore, the tuning process identified a valid candidate configuration, but the results do not suggest that a much larger hyperparameter search is necessary at this stage.

---

## 7. Overall Validation Results

Across the complete validation dataset, XGBoost produced:

| Metric | Result |
|---|---:|
| Precision | **0.5963** |
| Recall | **0.4591** |
| F1 | **0.5188** |
| Balanced Accuracy | **0.7170** |
| PR-AUC | **0.5565** |
| ROC-AUC | **0.9176** |
| Brier Score | **0.0506** |
| Actual Peak Rate | **7.48%** |
| Predicted Peak Rate | **5.76%** |
| Validation observations | **2,519,424** |

### Technical Interpretation

The model demonstrates useful discrimination between Peak and No-Peak observations.

PR-AUC of approximately **0.56** is substantially higher than the overall Peak prevalence of approximately 7.5%, which indicates meaningful ranking ability in an imbalanced classification problem.

ROC-AUC is approximately **0.92**.

At the common probability threshold of 0.50, XGBoost detects more Peaks than a highly conservative classifier would, but still misses a substantial proportion of actual Peaks.

### Non-Technical Interpretation

The model is able to identify periods that are more likely to become Peak hours.

However, using 0.50 as the alert cutoff means that some genuinely high-risk periods still remain below the final classification threshold.

---

## 8. Validation Fold Performance

Performance differs considerably between the two validation folds.

| Fold | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | **0.7385** | **0.3260** | **0.4524** | **0.6124** | **0.9153** | **0.0719** |
| 2024 | **0.4935** | **0.8219** | **0.6167** | **0.6709** | **0.9785** | **0.0294** |

The actual Peak prevalence also changes substantially:

- 2023: approximately **10.97%**
- 2024: approximately **4.01%**

### Interpretation

The model behaves very differently across time.

In 2023 it is relatively conservative:

- high precision;
- low recall.

In 2024 it becomes substantially more sensitive:

- lower precision;
- very high recall;
- higher F1;
- higher PR-AUC;
- considerably better calibration.

This indicates that temporal distribution changes affect the probability scale and the behavior of a fixed classification threshold.

---

## 9. Performance by FSA

The model shows substantial geographic variation.

### Fold 2023

| FSA | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| L4T | 0.3720 | 0.5153 | 0.4321 | 0.4267 |
| M5R | 0.4923 | 0.5601 | 0.5240 | 0.5336 |
| M5S | 0.3888 | 0.7092 | 0.5022 | 0.5915 |
| M6G | 0.7137 | 0.4711 | 0.5675 | 0.6418 |
| M9R | 0.5064 | 0.5354 | 0.5205 | 0.5416 |
| **M9W** | **0.9833** | **0.2689** | **0.4223** | **0.9143** |

### Fold 2024

| FSA | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| L4T | 0.5210 | 0.8544 | 0.6473 | 0.7339 |
| M5R | 0.3541 | 0.8526 | 0.5004 | 0.6008 |
| M5S | 0.3902 | 0.7766 | 0.5194 | 0.5337 |
| M6G | 0.4513 | 0.8566 | 0.5911 | 0.6752 |
| M9R | 0.5158 | 0.7857 | 0.6228 | 0.6642 |
| **M9W** | **0.5491** | **0.8159** | **0.6564** | **0.7012** |

### Interpretation

The model becomes much more balanced spatially in 2024.

M9W remains the most important structural stress case because its 2023 Peak prevalence is fundamentally different from the other FSA-period combinations.

---

## 10. M9W-2023 Structural Stress Case

M9W-2023 again emerges as the most unusual part of the validation dataset.

Its observed Peak Rate is approximately:

> **51.40%**

This is dramatically higher than the other FSAs in the same validation fold.

For M9W-2023, XGBoost produced:

| Metric | Result |
|---|---:|
| Precision | **0.9833** |
| Recall | **0.2689** |
| F1 | **0.4223** |
| PR-AUC | **0.9143** |
| ROC-AUC | **0.9106** |
| Brier Score | **0.3241** |

### Interpretation

The very high PR-AUC shows that XGBoost still ranks M9W observations effectively.

However, the model identifies only about 27% of the actual Peaks at threshold 0.50.

The main problem is therefore not an absence of predictive signal. It is a major mismatch between the probability distribution learned from historical data and the changed 2023 Peak regime.

This is consistent with the structural change previously identified for M9W during earlier phases of the project.

---

## 11. Impact of M9W-2023 on Global Performance

The additional diagnostic analysis quantified how strongly M9W-2023 affects the global results.

| Segment | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| All validation data | 0.5963 | 0.4591 | 0.5188 | 0.5565 | 0.9176 | 0.0506 |
| **Excluding M9W-2023** | **0.4977** | **0.7129** | **0.5862** | **0.6179** | **0.9671** | **0.0257** |
| M9W-2023 only | 0.9833 | 0.2689 | 0.4223 | 0.9143 | 0.9106 | 0.3241 |

### Interpretation

M9W-2023 strongly reduces the pooled recall, F1, and probability calibration.

After excluding that stress case for diagnostic purposes:

- recall rises from approximately **0.46 to 0.71**;
- F1 rises from approximately **0.52 to 0.59**;
- PR-AUC increases;
- ROC-AUC increases;
- Brier Score improves substantially.

M9W-2023 should **not** be removed from the official evaluation.

Instead, it should remain documented as a real temporal-generalization stress test.

---

## 12. Macro-Average Performance Across FSAs

Macro-averaging gives each FSA equal importance.

| Fold | Macro Precision | Macro Recall | Macro F1 | Macro PR-AUC | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 2023 | **0.5761** | **0.5100** | **0.4948** | **0.6082** | **0.9490** |
| 2024 | **0.4636** | **0.8236** | **0.5896** | **0.6515** | **0.9790** |

### Interpretation

The macro results confirm the fold-level conclusion.

XGBoost becomes much more sensitive to Peak events in 2024 while maintaining strong ranking performance across FSAs.

This reinforces that the change between folds is not caused only by different FSA sample sizes.

---

## 13. Threshold Analysis

The official comparison threshold remains:

> **0.50**

For the pooled validation dataset, the highest F1 among the evaluated thresholds occurs around **0.30**.

Representative values are:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.10 | 0.4195 | 0.6609 | 0.5133 |
| 0.20 | 0.4803 | 0.5928 | 0.5307 |
| **0.30** | **0.5230** | **0.5431** | **0.5328** |
| 0.40 | 0.5607 | 0.5003 | 0.5288 |
| **0.50** | **0.5963** | **0.4591** | **0.5188** |

The pooled F1 improvement from changing 0.50 to 0.30 is relatively modest.

---

## 14. Threshold Stability by Validation Fold

The additional threshold diagnostic revealed a very important temporal difference.

| Fold | F1-optimal diagnostic threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **2023** | **0.05** | 0.5202 | 0.6273 | **0.5687** |
| **2024** | **0.70** | 0.5837 | 0.6956 | **0.6347** |

### Interpretation

The F1-optimal probability threshold changes from approximately **0.05** in 2023 to approximately **0.70** in 2024.

This is a very large difference.

Therefore, a single threshold optimized on pooled validation observations would hide substantial temporal calibration instability.

This is stronger evidence that:

> **classification threshold selection must remain separate from model selection.**

The current 0.50 threshold should continue to be used for fair candidate-model comparison.

An operational threshold should only be selected later and should explicitly consider temporal robustness, calibration, and the relative cost of missed Peaks versus false alarms.

---

## 15. Performance by Forecast Horizon

XGBoost retains useful Peak-Risk information throughout the 24-hour prediction window.

The strongest performance generally occurs at the shortest horizons.

For example, during 2024:

| Horizon | Precision | Recall | F1 |
|---:|---:|---:|---:|
| h+1 | **0.6667** | **0.8666** | **0.7536** |
| h+6 | 0.5603 | 0.8533 | 0.6764 |
| h+12 | 0.4674 | 0.7973 | 0.5894 |
| h+18 | 0.4492 | 0.8197 | 0.5804 |
| h+24 | **0.4252** | **0.8207** | **0.5601** |

### Interpretation

As the prediction horizon increases:

- precision generally decreases;
- recall remains high in 2024;
- F1 gradually declines.

This means that longer-horizon XGBoost predictions tend to generate more false-positive alerts while still capturing many actual Peak events.

The distinction between short- and long-horizon performance should therefore be retained during final model comparison.

---

## 16. Feature Importance

The strongest model-native feature importances were:

| Feature | Importance |
|---|---:|
| Target hour sine | **0.1833** |
| Target hour | **0.1293** |
| Summer season | **0.0595** |
| Temperature at forecast origin | **0.0484** |
| Demand lag 24h | **0.0481** |
| Target month cosine | 0.0443 |
| Target month sine | 0.0410 |
| Winter season | 0.0383 |
| M9W indicator | 0.0372 |
| Target hour cosine | 0.0370 |
| M5R indicator | 0.0358 |
| M5S indicator | 0.0274 |
| L4T indicator | 0.0258 |
| Weekend indicator | 0.0233 |
| Spring season | 0.0213 |

### Interpretation

XGBoost relies on a combination of:

- **time of day**;
- **season**;
- **temperature**;
- **recent electricity demand**;
- **geographic context**.

This is conceptually consistent with the Peak-Risk problem.

Peak events depend strongly on when demand occurs, the seasonal context, weather conditions, recent consumption, and the specific FSA.

Model-native feature importance should not be interpreted as causality.

If XGBoost becomes a finalist, SHAP analysis could later provide more detailed global and local explanations.

---

## 17. Probability Calibration

The Brier Score changes substantially across validation contexts.

Global Brier Score:

> **0.0506**

2023:

> **0.0719**

2024:

> **0.0294**

M9W-2023:

> **0.3241**

Excluding M9W-2023:

> **0.0257**

### Interpretation

XGBoost has useful discriminative power, but its probability calibration is sensitive to temporal regime change.

The large deterioration for M9W-2023 explains why the same fixed threshold behaves so differently between validation folds.

This confirms that discrimination and calibration must be evaluated separately.

---

## 18. Main Strengths

The XGBoost Peak-Risk candidate demonstrates several strengths:

- strong overall discrimination;
- useful PR-AUC relative to Peak prevalence;
- high recall during the 2024 validation period;
- meaningful predictive signal throughout all 24 horizons;
- effective nonlinear modeling of calendar, weather, historical-demand, and FSA interactions;
- explicit treatment of class imbalance;
- training-only Peak threshold construction;
- leakage-safe preprocessing;
- standardized outputs suitable for direct candidate comparison;
- strong ranking ability even in the difficult M9W-2023 stress case.

---

## 19. Main Limitations

The evaluation identifies several limitations:

1. Global PR-AUC and ROC-AUC are useful but not dominant enough to justify declaring XGBoost the final model.
2. Probability calibration changes substantially between 2023 and 2024.
3. M9W-2023 strongly influences pooled performance.
4. The F1-optimal threshold is extremely unstable across validation folds.
5. Precision decreases noticeably at longer horizons in 2024.
6. FSA performance remains heterogeneous.
7. Model-native feature importance is not causal.
8. Additional tuning may produce incremental gains, but calibration and temporal stability currently appear more important than model capacity.

---

## 20. Hyperparameter and Threshold Decision

### Hyperparameter Tuning

**No additional hyperparameter tuning is recommended at this stage.**

The moderate search produced a valid configuration, and the two strongest tuning alternatives were already very close.

Further tuning should only be reconsidered if XGBoost becomes a finalist or near-finalist after comparison with the remaining Peak-Risk candidates.

### Classification Threshold

**The official comparison threshold remains 0.50.**

The additional diagnostic shows that an F1-optimal threshold is not temporally stable:

- approximately 0.05 in 2023;
- approximately 0.70 in 2024.

Therefore, choosing one optimized threshold now would risk overfitting the validation periods.

Operational threshold selection should be performed later as a separate decision.

---

## 21. Current Model Decision

**Status: Accepted as a valid Peak-Risk candidate model.**

The XGBoost Classifier is technically complete and ready for later model comparison.

No additional retraining, hyperparameter search, or threshold optimization is required before continuing to the next Peak-Risk candidate.

The final 2025 holdout remains protected.

---

## 22. Modeling Implications

The XGBoost experiment provides several important conclusions for the Peak-Risk workflow:

- Peak-Risk contains strong nonlinear predictive signal.
- Time of day is one of the strongest determinants of model risk ranking.
- Season and temperature provide meaningful additional predictive information.
- Historical demand remains relevant but does not dominate the model completely.
- PR-AUC should remain the main candidate-selection metric.
- Probability calibration should be considered alongside discrimination.
- M9W-2023 should remain a formal structural stress case.
- Threshold selection cannot be treated as a simple global optimization problem.
- Short- and long-horizon classification behavior should be compared separately.
- Final model selection should consider PR-AUC, Recall, Precision, F1, calibration, FSA stability, horizon stability, and temporal robustness together.

---

## 23. Supporting Evidence

Detailed outputs and diagnostic visualizations are available in the XGBoost branch, particularly:

- `XGBC_03_training.ipynb`
- `XGBC_04_hyperparameter_tuning.ipynb`
- `XGBC_05_evaluation.ipynb`
- `XGBC_06_interpretation.ipynb`
- `XGBC_04_tuning_details.csv`
- `XGBC_04_tuning_summary.csv`
- `XGBC_05_global_metrics.csv`
- `XGBC_05_fold_metrics.csv`
- `XGBC_05_fsa_metrics.csv`
- `XGBC_05_horizon_metrics.csv`
- `XGBC_05_class_balance.csv`
- `XGBC_05_threshold_analysis.csv`
- `XGBC_06_feature_importance.csv`

Recommended figures to retain as primary evidence are:

1. **Precision-Recall Curves**
2. **Confusion Matrices**
3. **PR-AUC by Forecast Horizon**
4. **Peak Rate by FSA**
5. **False Negative Rate by FSA**
6. **Threshold Diagnostics by Validation Fold**
7. **Calibration Curves**
8. **Top Feature Importances**

Together, these figures summarize discrimination, class imbalance, horizon behavior, spatial differences, temporal calibration, threshold instability, and predictor importance.
