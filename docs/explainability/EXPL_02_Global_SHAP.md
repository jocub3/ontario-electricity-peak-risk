# EXPL_02 — Global SHAP Analysis

## Purpose

SHAP was calculated using a reproducible sample and representative forecast
horizons to limit memory pressure from the large horizon-specific Random Forest
artifacts.

## Mean absolute SHAP ranking

| task   | feature                        | feature_family   |   mean_abs_shap |
|:-------|:-------------------------------|:-----------------|----------------:|
| rf     | target_lag_24h                 | demand_history   |     2719.79     |
| rf     | target_lag_48h                 | demand_history   |      487.888    |
| rf     | target_lag_168h                | demand_history   |      426.996    |
| rf     | origin__Temp (°C)              | weather          |      196.705    |
| rf     | origin_rolling_mean_24h        | demand_history   |      126.566    |
| rf     | origin__reported_premise_count | context          |       58.1635   |
| rf     | origin_rolling_std_24h         | demand_history   |       48.8168   |
| rf     | target_hour                    | calendar_time    |       40.5693   |
| rf     | origin__Stn Press (kPa)        | weather          |       31.8702   |
| rf     | fsa                            | spatial          |       30.2535   |
| rf     | target_hour_sin                | calendar_time    |       29.6863   |
| rf     | target_hour_cos                | calendar_time    |       28.6252   |
| rf     | target_weekday_sin             | calendar_time    |       27.49     |
| rf     | target_weekday                 | calendar_time    |       22.5931   |
| rf     | origin_rolling_mean_168h       | demand_history   |       22.2716   |
| rf     | target_month_cos               | calendar_time    |       16.9537   |
| rf     | target_weekday_cos             | calendar_time    |       14.4201   |
| rf     | origin__Rel Hum (%)            | weather          |       13.7373   |
| rf     | target_season                  | calendar_time    |       11.3342   |
| rf     | target_is_weekend              | calendar_time    |        8.6097   |
| rf     | target_month_sin               | calendar_time    |        7.9541   |
| rf     | target_month                   | calendar_time    |        7.35271  |
| xgb    | target_hour                    | calendar_time    |        1.47115  |
| xgb    | origin__Temp (°C)              | weather          |        1.4587   |
| xgb    | target_lag_24h                 | demand_history   |        1.43008  |
| xgb    | target_hour_sin                | calendar_time    |        0.875943 |
| xgb    | fsa                            | spatial          |        0.756503 |
| xgb    | target_month_cos               | calendar_time    |        0.686711 |
| xgb    | origin__reported_premise_count | context          |        0.511926 |
| xgb    | origin_rolling_mean_24h        | demand_history   |        0.496087 |
| xgb    | target_month_sin               | calendar_time    |        0.309902 |
| xgb    | origin_rolling_std_24h         | demand_history   |        0.302512 |
| xgb    | target_weekday                 | calendar_time    |        0.291441 |
| xgb    | origin_rolling_mean_168h       | demand_history   |        0.27442  |
| xgb    | target_hour_cos                | calendar_time    |        0.251577 |
| xgb    | target_lag_168h                | demand_history   |        0.226772 |
| xgb    | target_weekday_cos             | calendar_time    |        0.214468 |
| xgb    | origin__Rel Hum (%)            | weather          |        0.204206 |
| xgb    | target_month                   | calendar_time    |        0.191856 |
| xgb    | origin__Stn Press (kPa)        | weather          |        0.190001 |

## Scope

SHAP values explain the fitted final models; they do not establish causality.
For the binary XGBoost classifier, local SHAP contributions are interpreted on
the model's native/raw output scale unless explicitly configured otherwise.
The operational Peak-Risk score is reported separately.
