# Logistic Regression Peak-Risk Baseline – Overview

## Purpose

Logistic Regression establishes the reference performance for the Peak-Risk classification task.

The purpose of the baseline is to provide a simple, interpretable Machine Learning model against which more flexible classifiers such as Random Forest, XGBoost, LightGBM, and HistGradientBoosting can later be compared.

## Conceptual Approach

The Peak-Risk problem asks whether each of the following 24 hours will be a Peak hour.

The baseline therefore trains one Logistic Regression classifier for each horizon:

`peak_h01 ... peak_h24`

The model uses a deliberately compact set of predictors:

- FSA;
- target season;
- cyclical target-hour, weekday, and month information;
- weekend indicator;
- electricity consumption at the same target hour 24 hours earlier;
- electricity consumption at the same target hour 168 hours earlier.

No observed future electricity demand is used as an input.

## Peak Definition

Peak labels follow the shared project rule:

- threshold based on the **97.5th percentile**;
- calculated separately by **FSA + season**;
- thresholds fitted using the training period only;
- frozen thresholds applied to the corresponding validation period.

This preserves the anti-data-leakage policy established during Target Definition and Modeling Foundation.

## Preprocessing and Training

Unlike Seasonal Naive, Logistic Regression is a learned model.

For every horizon, the model is trained using:

- numeric imputation and standardization;
- categorical imputation and one-hot encoding;
- Logistic Regression from `scikit-learn`.

The preprocessing parameters and model coefficients are learned from the training data only.

Two development experiments are performed:

- **Fold 2023:** train using the historical period before 2023 and validate on 2023.
- **Fold 2024:** expand the training history and validate on 2024.

The final 2025 holdout remains untouched.

No hyperparameter search is performed because this implementation is intentionally a fixed baseline.

## Main Results

### Overall Development Performance

| Metric | Result |
|---|---:|
| Precision | 0.843 |
| Recall | 0.306 |
| F1 | 0.450 |
| PR-AUC | 0.675 |
| ROC-AUC | 0.944 |
| Balanced Accuracy | 0.651 |

### Performance by Validation Fold

| Fold | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| 2023 | 0.899 | 0.322 | 0.474 | 0.744 |
| 2024 | 0.700 | 0.265 | 0.384 | 0.539 |

Performance is clearly stronger in 2023 than in 2024.

The Peak prevalence also changes substantially:

- approximately **10.97%** during the 2023 fold;
- approximately **4.01%** during the 2024 fold.

This confirms that the classification problem is temporally non-stationary.

### Performance by FSA

Performance differs considerably by FSA.

- **M9W** shows the strongest PR-AUC in both folds, but it also has a substantially higher Peak rate, especially in 2023.
- **M5S** produces no positive classifications at the default probability threshold of 0.50, resulting in Precision, Recall, and F1 equal to zero.
- However, M5S still has non-zero PR-AUC, which means the predicted probabilities retain some ranking information even though none cross the initial 0.50 classification threshold.

This difference reinforces the importance of evaluating Peak-Risk models by FSA rather than only through one global metric.

### Performance by Forecast Horizon

The metrics remain very similar across the 24 forecast horizons.

This indicates that, for the current baseline predictor set, predictive performance is not strongly deteriorating between `h+1` and `h+24`.

## Interpretation

### Technical Interpretation

The baseline achieves an overall **PR-AUC of 0.675**, which is the project's primary Peak-Risk model-selection metric.

Precision is high at approximately **84%**, while Recall is much lower at approximately **31%**.

This means the default classifier is conservative: when it predicts a Peak, that prediction is often correct, but many real Peak hours are not identified.

The large difference between the 2023 and 2024 folds indicates that temporal stability will be an important criterion when evaluating candidate models.

### Non-Technical Interpretation

The model is relatively cautious.

When it raises a Peak warning, the warning is usually credible. However, it misses many Peak events.

Therefore, future models should aim not only to improve the overall score, but also to detect a larger proportion of real Peaks without producing too many false alarms.

A successful candidate model should ideally improve:

- PR-AUC;
- Recall;
- F1;
- consistency across different FSAs and validation periods.

## Threshold and Tuning Decision

The initial probability threshold remains fixed at **0.50** for the baseline comparison.

It should not be optimized inside this baseline branch because all classifiers need a common reference point.

No hyperparameter tuning is performed.

Later model-selection work may evaluate alternative operational probability thresholds using validation data only.

## Conclusion

**Accepted as the Peak-Risk baseline.**

The results are sufficient to serve as the reference benchmark for candidate Peak-Risk classifiers.

The baseline also exposes two important challenges that future models should address:

1. relatively low Recall;
2. substantial variation in performance across FSAs and temporal folds.

