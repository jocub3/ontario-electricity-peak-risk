# EXPL_04 / EXPL_06 — Local Prediction Explanation

Forecast origin: 2026-03-25 23:00:00

FSA: M9W

Target: 2026-03-26 17:00:00

RF forecast: 15,924.02 kWh

XGB Peak-Risk score: 0.7589

Official alert threshold: 0.06

## Forecasting contribution table

| task   |   horizon | feature                        | feature_value       |   shap_contribution |   abs_contribution | direction          |
|:-------|----------:|:-------------------------------|:--------------------|--------------------:|-------------------:|:-------------------|
| rf     |        18 | target_lag_24h                 | 15886.948577        |           5555.69   |          5555.69   | increases forecast |
| rf     |        18 | target_lag_48h                 | 15969.967764        |            954.29   |           954.29   | increases forecast |
| rf     |        18 | target_lag_168h                | 15730.545153        |            630.363  |           630.363  | increases forecast |
| rf     |        18 | origin__Temp (°C)              | 2.7                 |            -72.3284 |            72.3284 | decreases forecast |
| rf     |        18 | origin__reported_premise_count | 11201               |             68.3145 |            68.3145 | increases forecast |
| rf     |        18 | origin_rolling_mean_24h        | 14209.05957775      |             60.6591 |            60.6591 | increases forecast |
| rf     |        18 | target_weekday_sin             | 0.43388373911755823 |            -44.8497 |            44.8497 | decreases forecast |
| rf     |        18 | origin_rolling_mean_168h       | 14422.61936907738   |             39.2293 |            39.2293 | increases forecast |
| rf     |        18 | fsa                            | M9W                 |             38.4692 |            38.4692 | increases forecast |
| rf     |        18 | origin__Stn Press (kPa)        | 99.0                |            -29.1684 |            29.1684 | decreases forecast |

## Peak-Risk contribution table

| task   |   horizon | feature                  | feature_value       |   shap_contribution |   abs_contribution | direction            |
|:-------|----------:|:-------------------------|:--------------------|--------------------:|-------------------:|:---------------------|
| xgb    |        18 | fsa                      | M9W                 |           -1.52608  |           1.52608  | decreases risk score |
| xgb    |        18 | target_lag_24h           | 15886.948577        |            1.41636  |           1.41636  | increases risk score |
| xgb    |        18 | target_month_sin         | 0.8660254037844386  |            0.784727 |           0.784727 | increases risk score |
| xgb    |        18 | target_month_cos         | 0.5000000000000001  |            0.609197 |           0.609197 | increases risk score |
| xgb    |        18 | target_hour_sin          | -0.9659258262890683 |            0.601483 |           0.601483 | increases risk score |
| xgb    |        18 | origin__Temp (°C)        | 2.7                 |           -0.533479 |           0.533479 | decreases risk score |
| xgb    |        18 | target_hour              | 17                  |            0.466554 |           0.466554 | increases risk score |
| xgb    |        18 | target_weekday_cos       | -0.900968867902419  |           -0.295688 |           0.295688 | decreases risk score |
| xgb    |        18 | origin_rolling_mean_168h | 14430.53597388095   |           -0.251184 |           0.251184 | decreases risk score |
| xgb    |        18 | target_season            | Spring              |            0.24904  |           0.24904  | increases risk score |

## Interpretation note

The explanation is local to the selected FSA, forecast origin, and target
horizon. A feature contribution is not a causal effect. For XGBoost, the
operational alert threshold is a separate decision rule applied after the model
produces its risk score.
