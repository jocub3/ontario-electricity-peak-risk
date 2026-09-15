# Decision-Support Application — Phase Overview

## Purpose

The application converts the validated forecasting, Peak-Risk,
explainability, and weather-sensitivity outputs into an interactive
decision-support interface.

## Architecture

The local academic pipeline retains the full frozen model artifacts.
The public Streamlit deployment is intentionally lightweight and reads
precomputed Parquet/JSON outputs only.

## Public Demo Coverage

- Forecast origins: **3**
- FSAs: **6**
- Forecast horizons: **24**
- Prediction rows: **432**
- Weather-scenario rows: **2,160**
- Official Peak-Risk threshold: **0.06**

## Main Views

1. Forecast Selection & Input Validation
2. Executive Summary
3. Forecast & Peak-Risk Monitor
   - demand heatmap
   - Peak-Risk heatmap
   - 24-hour demand view
   - risk timeline
4. Weather Sensitivity & Scenario Comparison
5. Model Insights
   - Explainability
   - Final Holdout performance
   - Model comparison
6. Project & Deployment Information

## Deployment Boundary

The public application does **not** load the Random
Forest artifact bundle. Model execution remains an offline/local pipeline
concern. The web application remains interactive because users can select
forecast origins, FSAs, horizons, and weather scenarios from the
precomputed validated dataset.

## Interpretation Boundaries

- Forecasts are direct h+1...h+24 predictions.
- Peak-Risk alerting uses the frozen 0.06 threshold.
- Weather scenarios perturb only forecast-origin temperature in Model v1.
- Scenario results are model sensitivity, not causal effects.
- Public demo results are precomputed from the validated frozen pipeline.

## Phase Status

**Decision-Support Application / Dashboard: READY FOR STREAMLIT DEPLOYMENT**
