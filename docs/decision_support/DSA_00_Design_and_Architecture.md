# DSA_00 — Design and Architecture

## Decision

The application uses a two-layer architecture:

1. **Local analytical/production layer**
   - operational demand and weather files;
   - IFB validation;
   - frozen 24-horizon RF forecasting;
   - frozen 24-horizon XGBoost Peak-Risk;
   - weather-sensitivity analysis;
   - precomputation of demo outputs.

2. **Public presentation layer**
   - Streamlit;
   - precomputed Parquet/JSON only;
   - no 13+ GB Random Forest bundle;
   - no retraining or model-selection logic.

## Navigation

The dashboard uses six top-level sections:

1. Forecast Selection
2. Executive Summary
3. Forecast & Peak-Risk
4. Weather Scenarios
5. Model Insights
6. Project Information

Heatmaps, 24-hour forecast, Peak-Risk monitor, explainability, performance,
model comparison, and project details are grouped inside these sections to
avoid excessive navigation.

## Input policy

Forecast origins are discovered from actual operational coverage. The
application does not accept arbitrary date/time text in public-demo mode.
The local pipeline remains responsible for validating the 168-hour observed
demand history and h0...h24 weather forecast grid.

## Deployment policy

The public application is interactive but not an online inference service.
Users can change forecast origin, FSA, and weather scenario using precomputed
validated outputs. Full model inference remains available locally.
