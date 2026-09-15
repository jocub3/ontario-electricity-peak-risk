# Random Forest Regressor --- Forecasting Candidate Model Overview

## 1. Purpose

The Random Forest Regressor was developed as a candidate model for the
24-hour electricity-demand forecasting task. Its objective is to improve
on the Seasonal Naive baseline while following the common Modeling
Foundation rules established for all forecasting models.

The model predicts electricity consumption independently for each
forecast horizon from **h+1 through h+24** and is evaluated using the
same temporal validation framework that will later allow a fair
comparison with the other forecasting candidates.

The final **2025 holdout was not evaluated** during this stage.

------------------------------------------------------------------------

## 2. Conceptual Modeling Process

The process can be understood as follows:

1.  Historical electricity-demand, calendar, weather, and contextual
    information available at the forecast origin is prepared.
2.  The model uses recent historical consumption patterns together with
    contextual variables to learn how future electricity demand behaves.
3.  A separate Random Forest prediction is produced for each of the 24
    forecast horizons.
4.  A moderate hyperparameter search is performed using development data
    only.
5.  The selected configuration is evaluated on the common 2023 and 2024
    validation folds.
6.  Performance is examined globally and by validation fold, FSA, and
    forecast horizon.
7.  Additional diagnostic analysis is performed using the saved
    validation predictions, including residual behavior and temporal
    error patterns.

Random Forest combines many decision trees and averages their
predictions. This allows the model to capture nonlinear relationships
and interactions without requiring numeric feature scaling.

------------------------------------------------------------------------

## 3. Predictor Information

The candidate model uses information that is available at the prediction
origin or is known from historical observations. The most influential
predictors found by the model were:

  |Feature              |   Importance|
  |:--------------------|------------:|
  |Consumption lag 24h  |       69.09%|
  |Consumption lag 48h  |       17.24%|
  |Consumption lag 168h |        7.58%|
  |Rolling mean 24h     |        1.58%|
  |Temperature          |        1.19%|

The three historical demand lags account for approximately **94% of the
model's total feature importance**.

### Interpretation

The model relies primarily on the recent historical behavior of
electricity consumption. This is reasonable for short-term electricity
forecasting because demand has strong daily and weekly persistence.

Weather variables are used by the model, but their marginal importance
is considerably lower than historical demand. Temperature is the most
influential weather predictor, while humidity and station pressure have
smaller contributions.

This does **not** demonstrate that weather is irrelevant. Part of the
weather-related demand behavior may already be indirectly represented in
recent consumption values. A dedicated ablation experiment would be
required to measure the incremental contribution of weather
independently.

------------------------------------------------------------------------

## 4. Hyperparameter Selection

A moderate tuning process compared four Random Forest configurations
using representative forecast horizons and the common development folds.

The selected configuration was:

  |Parameter                |    Selected value |
  |:------------------------|------------------:|
  |Number of trees          |               160 |
  |Maximum depth            |              None |
  |Minimum samples per leaf |                 5 |
  |Maximum features         |               0.7 |

The selected configuration achieved the lowest mean MAE among the tested
alternatives.

No additional tuning was performed at this stage. The current model
already provides a meaningful improvement over the baseline, and further
refinement is deferred until all candidate models can be compared.

------------------------------------------------------------------------

## 5. Overall Validation Results

Across the complete development validation predictions, Random Forest
obtained:

  |Metric                 |             Result |
  |:----------------------|-------------------:|
  |MAE                    |     **542.26 kWh** |
  |RMSE                   |     **926.18 kWh** |
  |MAPE                   |          **5.55%** |
  |WAPE                   |          **5.96%** |
  |Bias                   |    **-125.31 kWh** |
  |Validation predictions |      **2,519,424** |

Compared with the Seasonal Naive baseline, which obtained an MAE of
approximately **612.05 kWh**, Random Forest reduces MAE by approximately
**11.4%**.

### Technical interpretation

The lower MAE, RMSE, MAPE, and WAPE indicate that Random Forest improves
the forecasting accuracy relative to the baseline. The negative bias
under the project metric convention (`predicted - actual`) indicates an
overall tendency to underpredict demand.

### Non-technical interpretation

The model performs better than simply assuming that tomorrow will behave
like the previous day. It has learned additional patterns from
historical demand and contextual information. However, it still tends to
predict slightly below the actual consumption level, particularly during
some high-demand periods.

------------------------------------------------------------------------

## 6. Validation Fold Performance

  |Fold  |   MAE (kWh) |  RMSE (kWh)  |   MAPE  |  WAPE  |  Bias (kWh) |
  |:-----|------------:|-------------:|--------:|-------:|------------:|
  |2023  |      572.53 |      992.60  |   5.74% |  6.34% |     -185.47 |
  |2024  |      512.06 |      854.83  |   5.37% |  5.58% |      -65.32 |

Performance improves in the 2024 validation fold. The model has both
lower error and less systematic underprediction in 2024.

The improvement is consistent with the expanding-window validation
design: the 2024 model has access to a longer historical training period
than the 2023 model.

------------------------------------------------------------------------

## 7. Performance Across FSAs

Model performance is not uniform across the six FSAs.

A particularly important result occurs in **M9W**:

-   2023 MAE: **1,188.72 kWh**
-   2023 MAPE: **7.53%**
-   2023 Bias: **-973.42 kWh**
-   2024 MAE: **767.71 kWh**
-   2024 MAPE: **4.96%**
-   2024 Bias: **-209.16 kWh**

M9W therefore represents the most important spatial challenge observed
in the Random Forest results. The large 2023 underprediction is
substantially reduced in 2024.

By contrast, M5S has the smallest absolute errors, with an MAE of
approximately **196 kWh** in both validation folds. Absolute errors
should not be compared across FSAs without considering differences in
demand scale; percentage metrics provide additional context.

### Interpretation

The M9W result confirms that a model can perform well globally while
experiencing much greater difficulty in a particular area or period. For
this reason, model selection should not be based only on the global MAE.

------------------------------------------------------------------------

## 8. Forecast-Horizon Behavior

Forecast accuracy generally decreases as the prediction horizon
increases.

For example, in the 2024 validation fold:

-   h+1 MAE is approximately **366 kWh**.
-   Error progressively increases across later horizons.
-   By the end of the 24-hour forecast window, the error is materially
    higher than at h+1.

This behavior is expected in multi-horizon forecasting. Information
available at the forecast origin is most useful for the nearest future
hours, while uncertainty increases as the model predicts further ahead.

Unlike the Seasonal Naive baseline, Random Forest therefore shows a
meaningful relationship between forecast distance and prediction error.

------------------------------------------------------------------------

## 9. Residual Analysis

Residuals were defined for the diagnostic analysis as:

`Residual = Actual - Predicted`

Therefore:

-   a positive residual indicates **underprediction**;
-   a negative residual indicates **overprediction**.

The residual summary is:

  
  |Fold  | Mean Residual | Median Residual | Residual Std. Dev. |  MAE   | P95 Absolute Error | Maximum  Absolute Error | 
  |:-----|--------------:|----------------:|-------------------:|-------:|-------------------:|------------------------:|  
  |2023  |        185.47 |        47.58    |             975.12 | 572.53 |         1,991.24   |              11,145.70  |
  |2024  |         65.32 |        17.05    |             852.33 | 512.06 |         1,794.72   |              10,127.57  |
  
------------------------------------------------------------------------

### Interpretation

Residuals confirm the underprediction already observed through the Bias
metric. The effect is stronger in 2023 than in 2024.

The median residual is much smaller than the mean residual in both
folds. Together with the large maximum errors, this indicates that most
predictions are substantially closer to the actual values than the most
extreme forecasting errors.

The P95 absolute error also improves from approximately **1,991 kWh in
2023** to **1,795 kWh in 2024**.

The residual plots should be retained as diagnostic evidence because
they show aspects of model behavior that are not visible from a single
global accuracy metric.

------------------------------------------------------------------------

## 10. Monthly and Seasonal Error Patterns

The validation analysis shows a clear temporal pattern in forecasting
error.

### 2023

Monthly MAE increases considerably during summer:

-   January: **408.97 kWh**
-   June: **752.70 kWh**
-   July: **1,185.43 kWh**
-   August: **857.56 kWh**
-   September: **695.38 kWh**

### 2024

A similar pattern appears:

-   January: **389.89 kWh**
-   June: **925.77 kWh**
-   July: **982.35 kWh**
-   August: **912.97 kWh**
-   September: **572.96 kWh**

Errors fall substantially again during October and November.

### Interpretation

The model has considerably greater difficulty during the summer period,
particularly from June through August. This pattern appears in both
validation years and therefore deserves attention when comparing Random
Forest with subsequent forecasting models.

The FSA × month analysis also shows that the summer increase is not
restricted to a single FSA, although the magnitude differs by area. M9W
contains particularly large errors during several high-error months.

This result provides a useful connection to the weather and seasonal
context of the forecasting problem, but it does not by itself establish
that temperature is the cause of the increased error.

------------------------------------------------------------------------

## 11. Validation Timeline Analysis

The full validation timeline compares aggregated Actual and Predicted
demand throughout each validation year.

Conceptually, this analysis is used to determine whether the model
follows the overall evolution of electricity consumption rather than
only producing acceptable average error metrics.

The accompanying residual timeline provides a complementary view by
showing periods in which predictions systematically move above or below
observed demand.

These visual diagnostics should be considered together with the
numerical metrics when evaluating model stability.

------------------------------------------------------------------------

## 12. Main Strengths

The Random Forest candidate demonstrates several positive
characteristics:

-   It clearly improves on the Seasonal Naive forecasting baseline.
-   It captures nonlinear relationships without requiring numeric
    standardization.
-   Performance improves in the 2024 validation fold.
-   It provides forecasts for the complete h+1 to h+24 horizon.
-   The model can identify the relative importance of its predictors.
-   Its main predictive signal is consistent with the strong temporal
    persistence found during earlier analysis.
-   The standardized outputs allow direct comparison with future
    forecasting candidates.

------------------------------------------------------------------------

## 13. Main Limitations and Observations

The evaluation also identifies several limitations:

1.  **Forecast accuracy decreases with horizon.**
2.  **The model tends to underpredict demand**, particularly in the 2023
    fold.
3.  **M9W presents substantially larger errors**, especially during
    2023.
4.  **Summer months show materially higher forecasting errors** in both
    validation years.
5.  The model depends strongly on recent historical consumption.
6.  Weather variables provide comparatively low marginal feature
    importance in the current model.
7.  Training and tuning are computationally expensive; the complete
    execution required more than 80 minutes in the development
    environment.
8.  Some extreme errors remain, with maximum absolute residuals above
    10,000 kWh.

These observations do not invalidate the model. Instead, they define the
behaviors that should be compared against subsequent forecasting
candidates.

------------------------------------------------------------------------

## 14. Current Model Decision

**Status: Accepted as a valid Forecasting candidate model.**

No additional tuning is performed at this stage.

The current Random Forest configuration provides a meaningful
improvement over the baseline and is sufficiently developed to proceed
to model comparison once the remaining candidate models are available.

Additional tuning may be reconsidered later only if Random Forest
becomes the best or one of the best-performing candidates.

The final 2025 holdout remains protected and has **not** been used to
evaluate or select this model.

------------------------------------------------------------------------

## 15. Modeling Implications

The Random Forest experiment provides several useful conclusions for the
next forecasting models:

-   Historical demand at 24h, 48h, and 168h is an exceptionally strong
    predictive signal.
-   Candidate models should be evaluated not only globally but also by
    FSA, horizon, and time period.
-   Reducing summer forecasting error should be an important point of
    comparison.
-   The behavior of M9W should remain a specific stability check.
-   Models that improve long-horizon accuracy without sacrificing
    short-horizon performance would represent a meaningful improvement.
-   Weather contribution should not be judged only through tree feature
    importance; its incremental value can be examined separately if
    required.
-   Final model selection should consider accuracy, stability,
    computational cost, and consistency across FSAs and horizons rather
    than only the lowest global MAE.

------------------------------------------------------------------------

## 16. Supporting Evidence

The detailed numerical results and visual diagnostics are available in
the Random Forest modeling outputs and notebooks, particularly:

-   `RF_03_training.ipynb`
-   `RF_04_evaluation.ipynb`
-   `RF_05_global_metrics.csv`
-   `RF_05_fold_metrics.csv`
-   `RF_05_fsa_metrics.csv`
-   `RF_05_horizon_metrics.csv`
-   `RF_06_feature_importance.csv`

Recommended figures to retain with the model documentation include:

1.  **Actual vs Predicted validation timeline**
2.  **Residual distribution / residual timeline**
3.  **Monthly validation MAE**
4.  **MAE by forecast horizon**
5.  **Feature importance**

These figures provide complementary evidence of overall fit, systematic
error, temporal stability, forecast-horizon degradation, and predictor
contribution.
