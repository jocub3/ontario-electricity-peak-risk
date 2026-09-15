# EXPL_01 — Global Feature Importance

## Purpose

This report documents built-in tree feature importance from the frozen final
Random Forest forecasting and XGBoost Peak-Risk artifacts. One-hot encoded
categorical variables are aggregated back to their public feature names.

## Global feature ranking

| task   | feature                        | feature_family   |   mean_importance_pct |   median_importance_pct |   min_importance_pct |   max_importance_pct |
|:-------|:-------------------------------|:-----------------|----------------------:|------------------------:|---------------------:|---------------------:|
| rf     | target_lag_24h                 | demand_history   |            69.6819    |              69.7202    |           68.2467    |           70.5275    |
| rf     | target_lag_48h                 | demand_history   |            17.6683    |              17.6892    |           16.8258    |           18.3977    |
| rf     | target_lag_168h                | demand_history   |             7.45763   |               7.43584   |            7.15172   |            7.80824   |
| rf     | origin_rolling_mean_24h        | demand_history   |             1.56709   |               1.4378    |            1.1767    |            2.26341   |
| rf     | origin__Temp (°C)              | weather          |             0.917428  |               0.621584  |            0.311911  |            2.94764   |
| rf     | origin_rolling_std_24h         | demand_history   |             0.496645  |               0.472886  |            0.372636  |            0.675992  |
| rf     | origin_rolling_mean_168h       | demand_history   |             0.350717  |               0.388788  |            0.198258  |            0.456273  |
| rf     | origin__reported_premise_count | context          |             0.315557  |               0.328154  |            0.232178  |            0.35073   |
| rf     | origin__Stn Press (kPa)        | weather          |             0.28991   |               0.311669  |            0.0781809 |            0.416156  |
| rf     | origin__Rel Hum (%)            | weather          |             0.166859  |               0.183306  |            0.0841148 |            0.195298  |
| rf     | target_hour                    | calendar_time    |             0.147577  |               0.14743   |            0.0832241 |            0.238935  |
| rf     | target_hour_cos                | calendar_time    |             0.143861  |               0.102363  |            0.0854591 |            0.289074  |
| rf     | fsa                            | spatial          |             0.129256  |               0.112397  |            0.0958369 |            0.217885  |
| rf     | target_weekday_sin             | calendar_time    |             0.11716   |               0.12548   |            0.0683188 |            0.139085  |
| rf     | target_weekday                 | calendar_time    |             0.111931  |               0.121566  |            0.0554783 |            0.13605   |
| rf     | target_hour_sin                | calendar_time    |             0.0891592 |               0.0892692 |            0.070173  |            0.131972  |
| rf     | target_weekday_cos             | calendar_time    |             0.0862548 |               0.0950887 |            0.0327151 |            0.114422  |
| rf     | target_month_cos               | calendar_time    |             0.078816  |               0.0818049 |            0.0485489 |            0.107254  |
| rf     | target_month_sin               | calendar_time    |             0.0639725 |               0.0713201 |            0.0291162 |            0.0785939 |
| rf     | target_month                   | calendar_time    |             0.0596313 |               0.0651484 |            0.0342054 |            0.0692637 |
| rf     | target_season                  | calendar_time    |             0.0381843 |               0.0390381 |            0.0243456 |            0.0525318 |
| rf     | target_is_weekend              | calendar_time    |             0.0221041 |               0.024283  |            0.0121973 |            0.0260289 |
| xgb    | fsa                            | spatial          |            20.3681    |              20.8344    |           15.1991    |           24.3779    |
| xgb    | target_hour                    | calendar_time    |            16.8408    |              16.8614    |           14.6007    |           19.538     |
| xgb    | target_season                  | calendar_time    |            15.5587    |              16.1325    |           12.2565    |           18.5361    |
| xgb    | target_hour_sin                | calendar_time    |            12.768     |              12.6097    |           11.4039    |           14.6549    |
| xgb    | origin__Temp (°C)              | weather          |             4.95814   |               4.18241   |            3.896     |            8.25891   |
| xgb    | target_lag_24h                 | demand_history   |             4.92946   |               5.18135   |            2.92071   |            6.32718   |
| xgb    | target_month_cos               | calendar_time    |             4.8331    |               4.59794   |            4.18102   |            7.00991   |
| xgb    | target_month_sin               | calendar_time    |             2.34388   |               2.32115   |            2.09955   |            2.67574   |
| xgb    | target_is_weekend              | calendar_time    |             2.20013   |               2.13944   |            0.90541   |            3.15793   |
| xgb    | target_hour_cos                | calendar_time    |             2.00702   |               2.06899   |            1.24832   |            2.66535   |
| xgb    | target_weekday_cos             | calendar_time    |             1.61546   |               1.53775   |            1.3845    |            1.91408   |
| xgb    | target_month                   | calendar_time    |             1.53606   |               1.46022   |            1.25298   |            2.06894   |
| xgb    | origin__reported_premise_count | context          |             1.5136    |               1.54278   |            1.24012   |            1.76363   |
| xgb    | target_weekday                 | calendar_time    |             1.39679   |               1.37172   |            1.24208   |            1.63566   |
| xgb    | origin_rolling_std_24h         | demand_history   |             1.23945   |               1.24464   |            0.977174  |            1.45446   |
| xgb    | origin_rolling_mean_168h       | demand_history   |             1.16246   |               1.20017   |            0.607029  |            1.45125   |
| xgb    | origin_rolling_mean_24h        | demand_history   |             1.07042   |               1.03559   |            0.668844  |            1.62252   |
| xgb    | origin__Rel Hum (%)            | weather          |             0.791884  |               0.774584  |            0.582376  |            1.039     |

## Feature-family summary

| task   | feature_family   |   importance_pct |
|:-------|:-----------------|-----------------:|
| rf     | demand_history   |        97.2223   |
| rf     | weather          |         1.3742   |
| rf     | calendar_time    |         0.958651 |
| rf     | context          |         0.315557 |
| rf     | spatial          |         0.129256 |
| xgb    | calendar_time    |        61.7765   |
| xgb    | spatial          |        20.3681   |
| xgb    | demand_history   |         9.84814  |
| xgb    | weather          |         6.49364  |
| xgb    | context          |         1.5136   |

## Interpretation policy

Built-in tree importance describes how much fitted trees use each transformed
predictor to reduce their training objective. It is not causal and should not
be interpreted as the direction of effect. SHAP analysis is used in the next
stage to add directional and local interpretation.
