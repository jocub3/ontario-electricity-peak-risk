# EXPL_03 — Weather Influence

## Purpose

This section isolates the contribution of forecast-origin temperature,
relative humidity, and station pressure in the frozen Model v1.

This is **model interpretation**, not weather scenario analysis. Model v1 uses
weather at the forecast origin; it does not use target-hour future weather as
a frozen predictor.

## Weather share of SHAP importance

| task   |   horizon |   total_mean_abs_shap |   weather_mean_abs_shap |   weather_share_pct |
|:-------|----------:|----------------------:|------------------------:|--------------------:|
| rf     |         1 |            4606.23    |               430.273   |             9.3411  |
| rf     |         6 |            4423.67    |               330.968   |             7.48174 |
| rf     |        12 |            4415.43    |               216.148   |             4.89528 |
| rf     |        18 |            4296.44    |               136.828   |             3.18469 |
| rf     |        24 |            4151.46    |                97.3487  |             2.34493 |
| xgb    |         1 |              12.3555  |                 2.60175 |            21.0574  |
| xgb    |         6 |              10.932   |                 1.9148  |            17.5155  |
| xgb    |        12 |              10.3641  |                 1.72647 |            16.6581  |
| xgb    |        18 |               9.56406 |                 1.53944 |            16.0961  |
| xgb    |        24 |               9.5695  |                 1.4821  |            15.4878  |

## Binned weather effects

| task   |   horizon | feature           | value_bin        |   observations |   mean_feature_value |   mean_shap |   mean_abs_shap |
|:-------|----------:|:------------------|:-----------------|---------------:|---------------------:|------------:|----------------:|
| rf     |         1 | origin__Temp (°C) | (-12.301, -6.42] |             24 |            -8.65833  |   240.642   |        242.574  |
| rf     |         1 | origin__Temp (°C) | (-6.42, -2.42]   |             24 |            -4.25     |   140.073   |        173.503  |
| rf     |         1 | origin__Temp (°C) | (-2.42, 1.17]    |             24 |            -0.766667 |    13.0823  |         91.3693 |
| rf     |         1 | origin__Temp (°C) | (1.17, 4.78]     |             24 |             2.85833  |  -307.998   |        309.344  |
| rf     |         1 | origin__Temp (°C) | (4.78, 8.0]      |             25 |             6.28     |  -359.436   |        359.436  |
| rf     |         1 | origin__Temp (°C) | (8.0, 13.68]     |             23 |            10.3304   |  -433.006   |        433.006  |
| rf     |         1 | origin__Temp (°C) | (13.68, 16.93]   |             24 |            15.425    |  -533.463   |        533.463  |
| rf     |         1 | origin__Temp (°C) | (16.93, 20.12]   |             24 |            18.6708   |  -395.737   |        395.737  |
| rf     |         1 | origin__Temp (°C) | (20.12, 23.02]   |             24 |            21.6833   |    73.1514  |        234.266  |
| rf     |         1 | origin__Temp (°C) | (23.02, 32.8]    |             24 |            26.5125   |  1350.3     |       1350.3    |
| rf     |         6 | origin__Temp (°C) | (-12.301, -6.42] |             24 |            -8.65833  |   173.298   |        175.378  |
| rf     |         6 | origin__Temp (°C) | (-6.42, -2.42]   |             24 |            -4.25     |    82.3674  |        147.062  |
| rf     |         6 | origin__Temp (°C) | (-2.42, 1.17]    |             24 |            -0.766667 |   -40.8715  |        116.877  |
| rf     |         6 | origin__Temp (°C) | (1.17, 4.78]     |             24 |             2.85833  |  -238.384   |        238.384  |
| rf     |         6 | origin__Temp (°C) | (4.78, 8.0]      |             25 |             6.28     |  -277.211   |        277.211  |
| rf     |         6 | origin__Temp (°C) | (8.0, 13.68]     |             23 |            10.3304   |  -313.914   |        313.914  |
| rf     |         6 | origin__Temp (°C) | (13.68, 16.93]   |             24 |            15.425    |  -381.553   |        381.553  |
| rf     |         6 | origin__Temp (°C) | (16.93, 20.12]   |             24 |            18.6708   |  -259.109   |        274.057  |
| rf     |         6 | origin__Temp (°C) | (20.12, 23.02]   |             24 |            21.6833   |    55.4423  |        139.684  |
| rf     |         6 | origin__Temp (°C) | (23.02, 32.8]    |             24 |            26.5125   |   919.038   |        919.038  |
| rf     |        12 | origin__Temp (°C) | (-12.301, -6.42] |             24 |            -8.65833  |   129.715   |        129.715  |
| rf     |        12 | origin__Temp (°C) | (-6.42, -2.42]   |             24 |            -4.25     |    62.5598  |         84.6415 |
| rf     |        12 | origin__Temp (°C) | (-2.42, 1.17]    |             24 |            -0.766667 |     5.39159 |         94.0293 |
| rf     |        12 | origin__Temp (°C) | (1.17, 4.78]     |             24 |             2.85833  |  -125.104   |        125.104  |
| rf     |        12 | origin__Temp (°C) | (4.78, 8.0]      |             25 |             6.28     |  -169.978   |        169.978  |
| rf     |        12 | origin__Temp (°C) | (8.0, 13.68]     |             23 |            10.3304   |  -179.042   |        179.042  |
| rf     |        12 | origin__Temp (°C) | (13.68, 16.93]   |             24 |            15.425    |  -202.24    |        202.24   |
| rf     |        12 | origin__Temp (°C) | (16.93, 20.12]   |             24 |            18.6708   |   -85.4419  |        117.504  |
| rf     |        12 | origin__Temp (°C) | (20.12, 23.02]   |             24 |            21.6833   |    97.0965  |        118.175  |
| rf     |        12 | origin__Temp (°C) | (23.02, 32.8]    |             24 |            26.5125   |   453.621   |        453.621  |
| rf     |        18 | origin__Temp (°C) | (-12.301, -6.42] |             24 |            -8.65833  |    28.4937  |         38.9786 |
| rf     |        18 | origin__Temp (°C) | (-6.42, -2.42]   |             24 |            -4.25     |     6.0118  |         29.9249 |
| rf     |        18 | origin__Temp (°C) | (-2.42, 1.17]    |             24 |            -0.766667 |   -16.4005  |         28.2148 |
| rf     |        18 | origin__Temp (°C) | (1.17, 4.78]     |             24 |             2.85833  |   -60.2893  |         60.2893 |
| rf     |        18 | origin__Temp (°C) | (4.78, 8.0]      |             25 |             6.28     |   -91.1399  |         91.1399 |
| rf     |        18 | origin__Temp (°C) | (8.0, 13.68]     |             23 |            10.3304   |   -86.6514  |         86.6514 |
| rf     |        18 | origin__Temp (°C) | (13.68, 16.93]   |             24 |            15.425    |   -98.0486  |        100.656  |
| rf     |        18 | origin__Temp (°C) | (16.93, 20.12]   |             24 |            18.6708   |   -61.3384  |         67.8074 |
| rf     |        18 | origin__Temp (°C) | (20.12, 23.02]   |             24 |            21.6833   |    21.8032  |         39.3738 |
| rf     |        18 | origin__Temp (°C) | (23.02, 32.8]    |             24 |            26.5125   |   185.123   |        185.123  |
| rf     |        24 | origin__Temp (°C) | (-12.301, -6.42] |             24 |            -8.65833  |    -7.8357  |         18.3133 |
| rf     |        24 | origin__Temp (°C) | (-6.42, -2.42]   |             24 |            -4.25     |    -9.42128 |         27.066  |
| rf     |        24 | origin__Temp (°C) | (-2.42, 1.17]    |             24 |            -0.766667 |    -2.40693 |         22.9948 |
| rf     |        24 | origin__Temp (°C) | (1.17, 4.78]     |             24 |             2.85833  |    -4.23955 |         18.1897 |
| rf     |        24 | origin__Temp (°C) | (4.78, 8.0]      |             25 |             6.28     |   -15.5956  |         19.8582 |
| rf     |        24 | origin__Temp (°C) | (8.0, 13.68]     |             23 |            10.3304   |   -22.7613  |         25.5052 |
| rf     |        24 | origin__Temp (°C) | (13.68, 16.93]   |             24 |            15.425    |    -7.68244 |         14.7573 |
| rf     |        24 | origin__Temp (°C) | (16.93, 20.12]   |             24 |            18.6708   |    23.5136  |         28.2971 |
| rf     |        24 | origin__Temp (°C) | (20.12, 23.02]   |             24 |            21.6833   |    61.2116  |         70.6857 |
| rf     |        24 | origin__Temp (°C) | (23.02, 32.8]    |             24 |            26.5125   |   -27.5998  |         86.2644 |

## Interpretation constraint

Observed SHAP patterns describe how the fitted models use weather jointly with
demand-history, calendar, FSA, and contextual variables. They are not causal
effects and should not be described as the result of changing weather while
holding every other process constant. Controlled perturbation belongs to the
later Weather Sensitivity phase.
