# Final Holdout Evaluation 2025 --- Overview

## Purpose

The Final Holdout Evaluation evaluates the two frozen finalist models on
the protected 2025 period. This is the first and final use of 2025 for
model evaluation. The purpose is to estimate how the selected system
performs on genuinely unseen data, without using the result to retune
hyperparameters, change features, redefine Peak-Risk, or modify the
operational classification threshold.

The evaluated models were:

-   **Forecasting:** Random Forest Regressor.
-   **Peak-Risk:** XGBoost Classifier with operational threshold
    **0.06**.

## Evaluation process

Both models were trained using the historical information available
before the 2025 holdout and then evaluated across the complete protected
period.

Forecasting performance was evaluated globally and by FSA, forecast
horizon, and month using MAE, RMSE, MAPE, WAPE, and Bias.

Peak-Risk performance was evaluated globally and by FSA, forecast
horizon, and month using Precision, Recall, F1, Balanced Accuracy,
PR-AUC, ROC-AUC, confusion-matrix counts, and Brier Score.

Diagnostic plots were also generated for forecasting errors and
Peak-Risk discrimination.

------------------------------------------------------------------------

## Random Forest Regressor --- 2025 results

### Global performance

  Metric         Result
  -------- ------------
  MAE        540.52 kWh
  RMSE       927.03 kWh
  MAPE            5.42%
  WAPE            5.59%
  Bias       -53.18 kWh

The model therefore produced an average absolute percentage error of
approximately **5.4%** across the final holdout. The negative bias
indicates a modest overall tendency to underpredict consumption.

Performance decreases progressively as the forecast horizon increases.
MAE rises from approximately **368.68 kWh at h+1** to approximately
**604.53 kWh at h+24**, which is expected because uncertainty increases
farther into the future.

Performance also varies by FSA. M5S obtained the lowest MAE (227.94
kWh), while M9W obtained the highest absolute MAE (751.67 kWh).
Percentage-based errors are more comparable across zones because
electricity-demand scale differs by FSA.

The strongest seasonal deterioration occurs during **June and July**,
with MAE of approximately 1,039 and 1,327 kWh respectively. July also
shows an overall negative bias of approximately -301 kWh. This indicates
that high-demand summer conditions remain more difficult for the
forecasting model.

### Interpretation

The final holdout confirms that Random Forest generalizes reasonably
well to unseen 2025 data. Its errors are not uniform: long forecast
horizons, some FSAs, and summer demand conditions are more challenging.
These patterns should be documented as limitations rather than used to
modify the frozen model.

------------------------------------------------------------------------

## XGBoost Classifier --- 2025 results

### Global performance

  Metric                    Result
  ----------------------- --------
  Operational threshold       0.06
  Precision                  0.341
  Recall                     0.930
  F1                         0.499
  Balanced Accuracy          0.898
  PR-AUC                     0.712
  ROC-AUC                    0.964
  Brier Score               0.0428

The classifier detected approximately **93.0% of actual Peak-Risk
cases**. This is consistent with the operational objective of
prioritizing the avoidance of missed peaks.

The trade-off is lower Precision (**34.1%**): the low operational
threshold intentionally generates more false alerts in exchange for high
Recall. The observed Peak rate was approximately **6.92%**, while the
model generated Peak alerts for approximately **18.87%** of
observations.

The Brier Score of **0.0428** indicates relatively low mean squared
probability error overall. It should be interpreted together with
discrimination and calibration evidence rather than as a standalone
measure.

Performance gradually deteriorates with forecast horizon. PR-AUC
decreases from approximately **0.857 at h+1** to **0.592 at h+24**,
while Brier Score increases from approximately **0.029 to 0.052**.
Recall nevertheless remains high throughout the 24-hour horizon.

There is meaningful variation by FSA. M6G achieved the strongest PR-AUC
(approximately 0.817), while M5S and M5R were comparatively more
difficult. M9W maintained very high Recall but generated a comparatively
high alert rate, which is consistent with the intentionally
recall-oriented threshold.

Monthly results also show that Peak-Risk behavior is not equally
difficult throughout the year. Months with very low underlying Peak
prevalence can produce low Precision even when Recall and ranking
performance remain useful.

### Interpretation

The final holdout supports XGBoost as a high-recall Peak-Risk detector.
It successfully identifies most actual peaks but intentionally accepts a
substantial number of false alerts. This behavior reflects the
previously selected operational objective: **missing a true Peak is
considered more costly than generating a false warning**.

------------------------------------------------------------------------

## Final assessment

The protected 2025 evaluation completed successfully and provides no
methodological reason to reopen model selection.

The two models exhibit complementary behavior:

-   Random Forest provides the numerical 24-hour electricity-demand
    forecast.
-   XGBoost provides the probability and operational alert for
    Peak-Risk.

The results reveal expected weaknesses---particularly longer horizons,
summer forecasting errors, and the Precision cost associated with the
high-Recall Peak-Risk strategy---but these are now **final out-of-sample
findings**, not reasons for further tuning.

Accordingly, the model configurations remain frozen after the 2025
evaluation.

## Decision

**Final Holdout Evaluation 2025: COMPLETE**

No further hyperparameter tuning, feature selection, Peak-definition
changes, or operational-threshold optimization should be performed using
the 2025 results.

The next phase is **Final Training**, where the frozen model
specifications are retrained using all permitted historical data and
serialized into the final deployable model artifacts.
