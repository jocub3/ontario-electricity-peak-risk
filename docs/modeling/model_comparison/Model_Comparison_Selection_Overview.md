# Model Comparison & Selection --- Overview

## 1. Purpose

This phase compared the completed Forecasting and Peak-Risk models under
the common rules established in Modeling Foundation. The objective was
not only to identify the model with the best pooled metric, but also to
review temporal stability, performance by FSA and forecast horizon, the
structural M9W-2023 case, and the operational meaning of the
classification errors.

The final 2025 holdout was not used during model comparison.

------------------------------------------------------------------------

## 2. Comparison Process

The comparison followed these general steps:

1.  Consolidate the standardized outputs produced independently by each
    model branch.
2.  Compare Forecasting models using the common regression metrics,
    especially MAE, RMSE, MAPE and WAPE.
3.  Compare Peak-Risk classifiers using PR-AUC, Precision, Recall, F1
    and Balanced Accuracy under the common comparison threshold of 0.50.
4.  Review performance by validation fold rather than relying only on
    pooled metrics.
5.  Review performance by FSA and across the 24 forecast horizons.
6.  Retain M9W-2023 as a structural stress case instead of removing it
    from the official evaluation.
7.  Consider temporal stability and the operational cost of False
    Negatives versus False Positives.
8.  Select the model that should advance to final refinement before the
    untouched 2025 holdout.

------------------------------------------------------------------------

## 3. Forecasting Comparison

### Main pooled results

  
  |Model                    | MAE (kWh)    | RMSE (kWh)   |   MAPE    |   WAPE    |
  |:------------------------|-------------:|-------------:|----------:|----------:|
  |Seasonal Naive           | 612.05       |  994.29      |    6.59%  |     6.73% |
  |Random Forest Regressor  |   **542.26** |   **926.18** | **5.55%** | **5.96%** |
  |XGBoost Regressor        |      606.27  |     1,106.77 |     5.93% |     6.66% |
  |LightGBM Regressor       |     580.84   |    1,044.37  |     5.74% |     6.38% |
                                               
  

Random Forest produced the strongest pooled forecasting performance. Its
MAE of approximately **542.26 kWh** represents an improvement of about
**11.4%** over the Seasonal Naive baseline.

### Temporal behavior

Random Forest also remained competitive in both development folds:

 | Fold   |     MAE  |   RMSE  |  MAPE  |  WAPE  |
 |:-------|---------:|--------:|-------:|-------:|
 | 2023   |  572.53  | 992.60  | 5.74%  | 6.34%  |
 | 2024   |  512.06  | 854.83  | 5.37%  | 5.58%  |

Performance improved in 2024. The model nevertheless showed systematic
underprediction in some high-demand conditions.

### FSA and M9W-2023

M9W-2023 remained the most important forecasting stress case. For Random
Forest, M9W-2023 produced an MAE of approximately **1,188.72 kWh**,
compared with substantially smaller errors in the other FSAs. The
difficulty was much smaller in M9W-2024.

The comparison therefore confirmed that global performance alone is
insufficient: geographic and temporal regime changes materially affect
forecast error.

### Forecasting selection

**Selected Forecasting model: Random Forest Regressor.**

It provided the best overall balance of MAE, RMSE, percentage error and
validation performance among the evaluated forecasting candidates.

------------------------------------------------------------------------

## 4. Peak-Risk Comparison

Peak-Risk is a strongly imbalanced classification problem. The overall
Peak prevalence across development validation is approximately
**7.48%**, so ordinary Accuracy is not an appropriate primary selection
metric.

The standardized comparison therefore emphasized:

-   PR-AUC;
-   Precision;
-   Recall;
-   F1;
-   Balanced Accuracy;
-   temporal stability;
-   FSA and horizon behavior;
-   False Negatives;
-   M9W-2023 behavior.

### Main comparison at threshold 0.50

  |Model                    |   Precision   |    Recall    |       F1   |    PR-AUC  |   Balanced Accuracy |
  |:------------------------|--------------:|-------------:|-----------:|-----------:|--------------------:|
  |Logistic Regression      |   **0.843**   |     0.306    |    0.450   | **0.675**  |     0.651           |
  |Random Forest Classifier |       0.669   |     0.355    |    0.464   |     0.579  |      0.671          |
  |XGBoost Classifier       |       0.596   | **0.459**    |**0.519**   |     0.557  |  **0.717**          |
  |LightGBM Classifier      |       0.555   | **0.470**    |    0.509   |     0.532  |        ---          |
                                                         
  

Logistic Regression achieved the highest pooled PR-AUC and Precision.
This is an important result: a baseline model can legitimately
outperform more complex models on a particular metric.

However, the operational objective changed the interpretation of the
comparison. Missing a real Peak was considered more costly than
producing an additional false alert. Under that criterion, Recall and
False Negatives became important guardrails alongside PR-AUC.

XGBoost provided the strongest overall operational compromise at the
common 0.50 threshold: it achieved the highest F1 among the four
finalists, substantially higher Recall than Logistic Regression, and
useful discrimination across the full 24-hour horizon.

------------------------------------------------------------------------

## 5. Temporal Stability

The classifiers behaved differently between the 2023 and 2024 validation
periods.

For XGBoost:

  |Fold   |  Precision |  Recall |      F1 |  PR-AUC|
  |------:|-----------:|--------:|--------:|--------|
  |2023   |     0.7385 |  0.3260 |  0.4524 |  0.6124|
  |2024   |     0.4935 |  0.8219 |  0.6167 |  0.6709|

This showed that the probability scale and fixed-threshold behavior were
not temporally stable, even though ranking performance remained useful.

The same general issue appeared in the other tree-based classifiers.
Therefore, the comparison distinguished between:

-   **discrimination/ranking quality**, represented by metrics such as
    PR-AUC; and
-   **operational classification behavior**, which depends on the
    selected probability threshold.

------------------------------------------------------------------------

## 6. M9W-2023 Stress Case

M9W-2023 was retained as an official structural stress case because its
Peak prevalence was approximately **51.4%**, far above the typical
development prevalence.

For XGBoost at the standardized 0.50 comparison threshold, M9W-2023
showed approximately:

-   Precision: **0.9833**
-   Recall: **0.2689**
-   F1: **0.4223**
-   PR-AUC: **0.9143**

The very high PR-AUC together with low Recall indicated that XGBoost
retained strong ranking information in this unusual regime, but the
fixed 0.50 threshold failed to convert much of that signal into Peak
alerts.

This result was one of the main reasons to separate **model selection**
from **operational threshold selection**.

------------------------------------------------------------------------

## 7. Final Selection Decision

### Forecasting

**Random Forest Regressor** was selected as the final Forecasting
candidate.

### Peak-Risk

The initial metric-based recommendation identified Logistic Regression
because it had the highest PR-AUC. After reviewing the operational
guardrails, particularly the higher cost assigned to missed Peaks,
**XGBoost Classifier** was selected to advance to the final Peak-Risk
refinement stage.

This does not mean Logistic Regression performed poorly. Instead, it
means that the final selection considered the operational objective in
addition to the primary discrimination metric.

------------------------------------------------------------------------

## 8. Decision Before Final Holdout

At the end of Model Comparison & Selection:

-   Forecasting finalist: **Random Forest Regressor**
-   Peak-Risk finalist: **XGBoost Classifier**
-   2025 final holdout: **still untouched**
-   XGBoost hyperparameters: **no further tuning planned**
-   Remaining Peak-Risk decision: **operational classification
    threshold**

The next Peak-Risk step was therefore a dedicated threshold-refinement
phase using development validation predictions only.
