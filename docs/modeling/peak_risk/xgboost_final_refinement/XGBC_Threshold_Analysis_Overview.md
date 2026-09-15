# XGBoost Peak-Risk --- Operational Threshold Analysis Overview

## 1. Purpose

After XGBoost was selected as the Peak-Risk finalist, a separate
refinement stage was performed to select the **operational probability
threshold** before opening the final 2025 holdout.

This stage was deliberately separated from hyperparameter tuning.

The XGBoost algorithm, feature set, preprocessing, Peak definition and
previously selected hyperparameters were kept unchanged. The only open
decision was the probability threshold used to convert predicted risk
probabilities into Peak / No-Peak alerts.

The final 2025 holdout remained untouched throughout this stage.

------------------------------------------------------------------------

## 2. Why the Threshold Required Separate Analysis

During candidate comparison, all classifiers used the common threshold:

`0.50`

This allowed a fair standardized comparison. However, the earlier
XGBoost diagnostics showed that the F1-optimal threshold changed
substantially between validation folds, indicating temporal instability
in the probability scale.

Because the operational objective considers **missing a real Peak more
costly than generating a false alert**, the final threshold needed to
prioritize Recall while still maintaining a minimum level of Precision.

------------------------------------------------------------------------

## 3. Threshold Analysis Process

The refinement followed these steps:

1.  Load the saved XGBoost development-validation probabilities.
2.  Sweep candidate thresholds from **0.02 to 0.90**.
3.  Calculate Precision, Recall, F1, Balanced Accuracy, False Negatives
    and false-alarm behavior at each threshold.
4.  Evaluate each threshold separately across the development folds.
5.  Apply operational guardrails:
    -   minimum Recall in every fold: **0.60**
    -   minimum Precision in every fold: **0.20**
6.  Among feasible thresholds, maximize the **worst-fold F1**.
7.  Use worst-fold Recall, mean F1 and mean Precision as additional
    tie-breaking criteria.
8.  Review the selected threshold by fold, FSA and forecast horizon.
9.  Re-evaluate M9W-2023 as a structural stress case.
10. Freeze the complete XGBoost configuration before the 2025 final
    holdout.

------------------------------------------------------------------------

## 4. Selected Operational Threshold

The selected threshold was:

> **0.06**

Operationally:

`Predicted probability >= 0.06 -> Peak-Risk`

`Predicted probability < 0.06 -> No Peak-Risk`

The threshold was not chosen manually. It was selected by the predefined
validation-only decision rule.

The threshold of 0.06 was near the operational boundary: increasing the
threshold further improved some Precision/F1 characteristics but caused
the worst-fold Recall to fall below the required 0.60.

------------------------------------------------------------------------

## 5. Selected-Threshold Results

### Global development performance

At threshold **0.06**, the model produced approximately:

  |Metric             |        Result|
  |:------------------|-------------:|
  |Precision          |    **38.14%**|
  |Recall             |    **70.60%**|
  |F1                 |    **49.52%**|
  |Balanced Accuracy  |    **80.67%**|
  |True Positives     |   **133,092**|
  |False Negatives    |    **55,425**|
  |False Positives    |   **215,901**|
  |False Alarm Rate   |     **9.26%**|

### Interpretation

The selected operating point deliberately increases sensitivity.

In practical terms, the model detects approximately **71 of every 100
actual Peaks** across the pooled development validation data. The cost
is lower Precision and a larger number of false alerts.

This trade-off is consistent with the project's operational assumption
that missing an actual Peak is more costly than producing an unnecessary
alert.

------------------------------------------------------------------------

## 6. Fold Stability

The selected threshold satisfies the minimum Recall requirement in both
validation folds, but its behavior differs materially across time.

  |Fold   |   Precision |      Recall |          F1|
  |:------|------------:|------------:|-----------:|
  |2023   |  **53.44%** |  **60.73%** |  **56.85%**|
  |2024   |      25.66% |  **97.52%** |      40.62%|

The threshold-selection summary records:

-   minimum fold Precision: **0.2566**
-   mean fold Precision: **0.3955**
-   minimum fold Recall: **0.6073**
-   mean fold Recall: **0.7913**
-   worst-fold F1: **0.4062**
-   mean fold F1: **0.4874**
-   worst-fold Balanced Accuracy: **0.7711**
-   mean Balanced Accuracy: **0.8498**
-   maximum missed-Peak rate across folds: **39.27%**
-   mean false-alarm rate: **9.16%**

### Interpretation

The threshold is operationally feasible, but the large difference
between 2023 and 2024 confirms a temporal distribution/calibration
shift.

In 2023 the model is comparatively balanced. In 2024 it becomes highly
sensitive and captures almost all Peaks, but produces more false alerts.

This temporal difference must remain documented as an important model
limitation.

------------------------------------------------------------------------

## 7. M9W-2023 Stress Case

M9W-2023 remained a dedicated stress case because its Peak prevalence is
approximately **51.4%**.

At threshold **0.06**, M9W-2023 produced approximately:

  |Metric              |     Result |
  |:-------------------|-----------:|
  |Precision           |  **93.08%**|
  |Recall              |  **55.45%**|
  |F1                  |  **69.50%**|
  |Balanced Accuracy   |  **75.54%**|

### Interpretation

The model is highly precise when it issues a Peak alert for M9W-2023,
but it still misses a substantial share of actual Peaks.

This confirms that M9W-2023 represents a structurally different
operating regime rather than a simple data error that should be removed.

------------------------------------------------------------------------

## 8. FSA and Horizon Review

The selected threshold was also evaluated across:

-   individual FSAs;
-   both development folds;
-   all forecast horizons from `h+1` through `h+24`.

The model retained useful Peak-detection ability across the complete
24-hour horizon. There was no evidence of a complete collapse at the
later horizons.

The FSA analysis also confirmed that the very high Recall observed in
2024 was not produced by a single geographic area; the high-sensitivity
behavior appeared broadly across the evaluated FSAs.

------------------------------------------------------------------------

## 9. Probability Calibration Observation

The low operational threshold should **not** be interpreted as meaning
that a predicted value of 0.06 is necessarily a perfectly calibrated 6%
real-world Peak probability.

The substantial change in threshold behavior between 2023 and 2024
indicates that probability calibration is temporally sensitive.

The model can still be used as a classifier/risk-ranking system, but any
future dashboard should avoid presenting the raw XGBoost probability as
a perfectly calibrated probability unless a dedicated calibration
analysis supports that interpretation.

------------------------------------------------------------------------

## 10. Frozen Final Configuration

After threshold selection, the Peak-Risk development configuration was
frozen as:

-   Algorithm: **XGBoostClassifier**
-   Operational threshold: **0.06**
-   Peak definition: **P97.5 by FSA + season, fitted on training data
    only**
-   Forecast horizons: **24**
-   Number of estimators: **400**
-   Maximum depth: **6**
-   Learning rate: **0.05**
-   Subsample: **0.80**
-   Column sample by tree: **0.80**
-   Minimum child weight: **5**
-   L2 regularization: **1.0**
-   Imbalance multiplier: **1.0**
-   Feature set: **unchanged**
-   Preprocessing: **unchanged**
-   Final 2025 holdout evaluated: **No**

The freeze manifest also records hashes of the development artifacts to
preserve reproducibility.

------------------------------------------------------------------------

## 11. Final Decision

**Status: Threshold analysis completed and XGBoost configuration frozen
for final holdout evaluation.**

The final Peak-Risk development configuration is therefore:

> **XGBoost Classifier + operational threshold 0.06**

No additional model tuning or threshold modification should be performed
before evaluating 2025.

The next step is the **Final Holdout Evaluation 2025**, using the frozen
configuration exactly as defined above. The result from 2025 must be
treated as a final out-of-sample evaluation rather than another
opportunity to modify the model.
