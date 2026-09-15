# Model Contract

All candidate models must use the same evaluation contract.

## Forecast output

Required columns:

- `origin_timestamp`
- `forecast_timestamp`
- `horizon`
- `actual`
- `prediction`
- `model_name`
- `region`

## Peak-risk output

Required columns:

- `forecast_timestamp`
- `actual_peak`
- `peak_probability`
- `predicted_peak`
- `model_name`
- `region`

## Shared rules

- Forecast horizon: 24 hours.
- Data splits must preserve chronological order.
- Hyperparameters and random seeds must be recorded.
- Metrics must be calculated on the same test periods.
- No future information may be used in features or target construction.

## Regions (`region` column)

Exactly two values, one model run per region (same 4 models, run twice — not 8 different models):

- `Downtown` — FSA M5S, M5R, M6G; weather from Toronto City station.
- `Airport-West` — FSA L4T, M9W, M9R; weather from Toronto Pearson station.

The peak-risk percentile threshold (`configs/default.yaml: peak_risk.percentile_threshold`) is computed
**separately per region**, on that region's training split only — the two regions have different consumption
scales and must not share a single global threshold.

## The 4 MVP models and their 24h forecasting strategy

| Role | Model | Strategy |
|---|---|---|
| Baseline | Seasonal-naive (`forecasting/baseline.py`) | Direct — value from the same hour N periods ago, computed independently per horizon step |
| Statistical | SARIMAX (`forecasting/sarimax.py`) | Recursive — native multi-step `statsmodels` forecast |
| ML | XGBoost (`forecasting/xgboost_model.py`) | Direct — a single model trained with horizon (1-24) as an input feature |
| Peak-risk classifier | XGBoost Classifier (`peak_risk/xgboost_classifier.py`) | N/A (classification, not sequential) |

`lightgbm_model.py`, `prophet_model.py`, `logistic_regression.py`, `random_forest.py`, and `catboost_classifier.py`
are out of scope for the MVP (see their docstrings) — kept as stubs for potential future comparison, not deleted.
