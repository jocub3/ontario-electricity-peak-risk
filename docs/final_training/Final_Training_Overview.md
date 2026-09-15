# Final Training --- Overview

## Purpose

The Final Training phase creates the definitive reusable model artifacts
after completion of the protected 2025 holdout evaluation. No model
selection, hyperparameter tuning, feature changes, Peak-Risk
redefinition, or threshold optimization is performed here.

## Final models

-   **Forecasting:** Random Forest Regressor.
-   **Peak-Risk:** XGBoost Classifier with the previously selected
    operational threshold of 0.06.

Because the project predicts 24 forecast horizons, one final model
artifact is produced for each horizon in each task.

## Process

The pipeline loads the frozen configurations and complete available
historical data, trains Random Forest independently for h+1 through
h+24, calculates the final FSA/season Peak thresholds, trains XGBoost
independently for h+1 through h+24, and serializes the trained pipelines
together with their metadata.

## Generated artifacts

The execution successfully generated:

-   **24 Random Forest model artifacts**
-   **24 XGBoost model artifacts**
-   Random Forest metadata
-   XGBoost metadata
-   Final `peak_thresholds.parquet`
-   Final model artifact manifest

Artifact validation confirmed that all 48 expected horizon-specific
model files exist. Both metadata files and the Peak threshold table were
also successfully validated.

## Result

**Status: FINAL_MODELS_TRAINED**

The protected 2025 holdout results were not used to alter the frozen
model configurations, preserving the methodological separation between
final evaluation and final training.

Final Training does not produce new predictive-performance metrics
because its purpose is deployment artifact creation rather than model
evaluation. Therefore, under the current design, it is expected that
`reports/final_training` contains no new performance reports.

## Final architecture

``` text
24 Random Forest forecasting models
                +
24 XGBoost Peak-Risk models
                +
Peak threshold table
                +
model metadata
```

These multiple internal artifacts will be exposed as one logical
prediction process during integration.

## Next phase

The next phase is **Integrated Prediction Pipeline**.

It will combine Forecasting and Peak-Risk so that each requested FSA and
each of the next 24 hours receives a standardized output containing:

-   forecast electricity consumption;
-   Peak-Risk probability/score;
-   Peak alert.

This integrated output will become the foundation for the project's
decision-support application and visualization layer.
