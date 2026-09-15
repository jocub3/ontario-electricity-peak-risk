# EXPL_04 / EXPL_06 — Local Prediction Explanation

Forecast origin: 2025-07-23 23:00:00

FSA: M9W

Target: 2025-07-24 17:00:00

RF demand forecast: 27,507.77 kWh

XGB Peak-Risk probability: 0.9448

Official Peak-Risk threshold: 0.06

## Forecasting contribution table

| task   |   horizon | feature                        | feature_value      |   shap_contribution |   abs_contribution | direction          |
|:-------|----------:|:-------------------------------|:-------------------|--------------------:|-------------------:|:-------------------|
| rf     |        18 | target_lag_24h                 | 24513.699999999997 |           12395     |          12395     | increases forecast |
| rf     |        18 | target_lag_48h                 | 20248.8            |            2113.79  |           2113.79  | increases forecast |
| rf     |        18 | origin_rolling_std_24h         | 5485.158704002664  |            1046.15  |           1046.15  | increases forecast |
| rf     |        18 | target_lag_168h                | 21929.3            |            1002.16  |           1002.16  | increases forecast |
| rf     |        18 | origin__Stn Press (kPa)        | 99.78              |             685.077 |            685.077 | increases forecast |
| rf     |        18 | origin_rolling_mean_24h        | 17679.3875         |             369.85  |            369.85  | increases forecast |
| rf     |        18 | origin__reported_premise_count | 11173              |             291.146 |            291.146 | increases forecast |
| rf     |        18 | origin_rolling_mean_168h       | 16919.842857142856 |             199.879 |            199.879 | increases forecast |
| rf     |        18 | fsa                            | M9W                |             148.759 |            148.759 | increases forecast |
| rf     |        18 | origin__Rel Hum (%)            | 82.0               |             134.148 |            134.148 | increases forecast |

## Peak-Risk contribution table

| task   |   horizon | feature                  | feature_value       |   shap_contribution |   abs_contribution | direction            |
|:-------|----------:|:-------------------------|:--------------------|--------------------:|-------------------:|:---------------------|
| xgb    |        18 | target_lag_24h           | 24513.699999999997  |            3.17254  |           3.17254  | increases risk score |
| xgb    |        18 | fsa                      | M9W                 |           -0.992464 |           0.992464 | decreases risk score |
| xgb    |        18 | origin_rolling_std_24h   | 5557.375755772668   |            0.926689 |           0.926689 | increases risk score |
| xgb    |        18 | target_month_cos         | -1.0                |           -0.836395 |           0.836395 | decreases risk score |
| xgb    |        18 | target_hour              | 17                  |            0.544708 |           0.544708 | increases risk score |
| xgb    |        18 | target_hour_sin          | -0.9659258262890683 |            0.543655 |           0.543655 | increases risk score |
| xgb    |        18 | target_weekday_cos       | -0.900968867902419  |           -0.326367 |           0.326367 | decreases risk score |
| xgb    |        18 | origin_rolling_mean_168h | 16949.061904761907  |           -0.258432 |           0.258432 | decreases risk score |
| xgb    |        18 | origin__Temp (°C)        | 20.8                |           -0.222152 |           0.222152 | decreases risk score |
| xgb    |        18 | target_season            | Summer              |           -0.203893 |           0.203893 | decreases risk score |

## Interpretation note

The explanation is local to the selected FSA, forecast origin, and target
horizon. A feature contribution is not a causal effect. For XGBoost, the
operational alert threshold is a separate decision rule applied after the model
produces its risk score.
