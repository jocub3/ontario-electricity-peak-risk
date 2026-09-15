# Explainability & Model Interpretation — Phase Overview

## Completed sections

- EXPL_00 Design and Scope
- EXPL_01 Global Feature Importance
- EXPL_02 Global SHAP
- EXPL_03 Weather Influence
- EXPL_04 Local Historical Explanations
- EXPL_05 Horizon Comparison
- EXPL_06 Operational Local Explanation

## Key findings

- RF top built-in features: target_lag_24h, target_lag_48h, target_lag_168h.
- RF top SHAP features: target_lag_24h, target_lag_48h, target_lag_168h.
- XGB top built-in features: fsa, target_hour, target_season.
- XGB top SHAP features: target_hour, origin__Temp (°C), target_lag_24h.
- RF mean weather share of SHAP importance across representative horizons: 5.45%.
- XGB mean weather share of SHAP importance across representative horizons: 17.36%.

## Methodological boundaries

- Explainability describes fitted-model behavior and is not causal inference.
- Model v1 weather interpretation applies to forecast-origin weather.
- SHAP sampling is intentionally memory-conscious because final Random Forest
  artifacts are large.
- The official Peak-Risk threshold remains a decision-policy layer separate
  from the XGBoost explanation.
- Weather perturbation scenarios are handled in the subsequent Weather
  Sensitivity / Scenario Analysis phase.
