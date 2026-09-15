# Operational Inference Feature Builder --- Phase Overview

## 1. Objective

The Operational Inference Feature Builder (IFB) converts data available
at a forecast origin into the exact feature schemas expected by the
frozen final models:

-   **Random Forest forecasting:** direct multi-horizon
    electricity-demand forecasts for h+1 through h+24.
-   **XGBoost Peak-Risk:** direct multi-horizon Peak-Risk scores for h+1
    through h+24.

The IFB is an inference and validation layer. It does **not** retrain
the models, alter their fitted parameters, redefine the target, or
invent missing historical consumption. The original design principle
remains unchanged: operational data must be transformed so that
inference reproduces the feature logic used during model development.

## 2. Final operational architecture

``` text
Historical project data (2021–2025)
              +
Incremental operational demand (2026+)
              +
Operational weather inputs
              ↓
Input loading and validation
              ↓
Recent-history window required at forecast origin
              ↓
Inference Feature Builder
              ↓
Frozen task-specific feature contracts
        ┌──────────────┴──────────────┐
        ↓                             ↓
24 Random Forest models        24 XGBoost models
(h+1 ... h+24 demand)          (h+1 ... h+24 risk)
        └──────────────┬──────────────┘
                       ↓
              144 integrated rows
              6 FSAs × 24 hours
                       ↓
          Forecast + Peak-Risk output
```

The final implementation uses **direct multi-horizon modeling**. Each
horizon has its own fitted artifact; therefore the 24 Random Forest
artifacts and 24 XGBoost artifacts are separate trained models, not
repeated calls to one fitted model.

## 3. Data architecture

Existing project history remains in the processed historical dataset.
Operational observations are added incrementally:

``` text
data/operational/
├── demand/
├── weather/
│   ├── history/
│   └── forecast/
└── templates/
```

Multiple 2026+ CSV files can be added over time. The loader:

-   discovers the incremental files;
-   concatenates them;
-   normalizes FSA and timestamps;
-   sorts records chronologically;
-   resolves overlapping FSA/timestamp keys using the latest operational
    record;
-   keeps historical project data separate from operational updates;
-   selects only the recent window required for a specific forecast
    origin during operational inference.

This avoids requiring one continuously rebuilt master CSV.

## 4. Required recent demand

For the frozen h+1...h+24 architecture, complete observed demand is
required from:

``` text
forecast_origin - 168h  →  forecast_origin
```

This supports target-relative demand lags at 24, 48 and 168 hours and
the origin rolling summaries.

The IFB validates both completeness and recency. If the required recent
demand window is incomplete, operational inference is blocked. Long
missing periods are **not** filled using recursive forecasts or
synthetic values.

Future consumption is never required as an input because it is the
quantity being forecast.

## 5. Weather contract

The operational interface accepts weather rows from h0 through h+24 and
validates the complete weather grid.

However, an important limitation of the frozen Model v1 was confirmed
during implementation: the final RF and XGBoost models use
**forecast-origin weather/context**, not target-hour future-weather
values as model features.

The frozen feature contract therefore uses:

-   `origin__Temp (°C)`
-   `origin__Rel Hum (%)`
-   `origin__Stn Press (kPa)`

Future h+1...h+24 weather is retained in the operational architecture
for validation, later sensitivity/scenario analysis, and the
decision-support layer. It is **not** silently inserted into Model v1
because doing so would change the model feature contract and require a
separately trained and validated model version.

## 6. Frozen feature contract

Both final model families use 22 public features:

``` text
target_hour
target_weekday
target_month
target_is_weekend
target_hour_sin
target_hour_cos
target_weekday_sin
target_weekday_cos
target_month_sin
target_month_cos
target_lag_24h
target_lag_48h
target_lag_168h
origin_rolling_mean_24h
origin_rolling_std_24h
origin_rolling_mean_168h
origin__Temp (°C)
origin__Rel Hum (%)
origin__Stn Press (kPa)
origin__reported_premise_count
fsa
target_season
```

The IFB creates the target timestamp and calendar/cyclical features for
every h+1...h+24 row, retrieves the required historical demand lags,
creates origin demand summaries, attaches origin weather and premise
count, and preserves FSA/season context.

## 7. Model-specific rolling semantics

Historical replay identified an important compatibility detail that must
be preserved in deployment.

Although the public rolling feature names are the same, the frozen final
model branches were trained with different origin-window semantics:

### Random Forest forecasting

The rolling window includes the observed consumption at the forecast
origin:

``` text
24h  : origin-23h  ... origin
168h : origin-167h ... origin
```

### XGBoost Peak-Risk

The rolling window uses observations strictly before the forecast
origin:

``` text
24h  : origin-24h  ... origin-1h
168h : origin-168h ... origin-1h
```

The IFB therefore constructs both internal variants and maps the
appropriate values to each frozen model immediately before inference. No
post-origin consumption is used.

This behavior was verified through historical replay rather than assumed
from column names.

## 8. Input and feature validation

The completed IFB validates:

-   operational file inventory;
-   duplicate FSA/timestamp keys;
-   required recent-demand coverage and recency;
-   complete h0...h+24 weather grid;
-   expected 6 FSAs;
-   h+1...h+24 forecast grid;
-   exact frozen RF feature contract;
-   exact frozen XGBoost feature contract;
-   missing and unsupported features;
-   all-null or partially-null model features;
-   out-of-domain inputs relative to historical development ranges;
-   historical replay equality against the original model-development
    feature builders.

For out-of-domain checks, unusual inputs are reported rather than
automatically replaced. This is especially relevant for weather values
near or outside the historical training domain.

## 9. Historical replay validation --- IFB_08

IFB_08 was executed with a forecast origin contained in the original
historical dataset:

``` text
2025-12-30 23:00:00
```

The replay compares IFB-generated features with the original RF and
XGBoost feature-building logic horizon by horizon.

After correcting the model-specific rolling semantics, historical replay
completed successfully. This establishes that the operational IFB can
reproduce the feature semantics expected by the frozen final models
before applying the system to post-holdout operational data.

## 10. Operational inference validation --- IFB_09

Operational inference was subsequently executed using:

``` text
forecast_origin = 2026-03-25 23:00:00
```

with 2026 operational inputs.

The integrated pipeline successfully produced:

``` text
6 FSAs × 24 horizons = 144 predictions
```

For every target hour, the output contains:

-   FSA;
-   forecast origin;
-   target timestamp;
-   horizon;
-   forecast electricity consumption;
-   Peak-Risk score;
-   Peak-Risk alert.

Final output validation checks include:

-   exactly 144 expected prediction rows;
-   unique prediction keys;
-   all six FSAs;
-   complete h+1...h+24 coverage for every FSA;
-   no missing forecasts or Peak-Risk scores;
-   positive demand forecasts;
-   Peak-Risk scores within \[0,1\];
-   consistency between `peak_risk_score`, the decision threshold, and
    `peak_alert`.

The 2026 operational simulation passed these checks.

## 11. Supplementary post-holdout evaluation

Because observed demand for the 24 target hours after the March 2026
forecast origin was available, IFB_09 also compared the generated
forecasts against subsequently observed demand.

This is treated as **supplementary post-holdout operational evidence**,
not as a new model-selection or tuning dataset.

The observed global forecasting results were approximately:

  Metric     Post-holdout result
  -------- ---------------------
  MAE                 188.45 kWh
  RMSE                247.60 kWh
  MAPE                     1.98%
  WAPE                     1.94%
  Bias                +72.22 kWh

These results provide additional evidence that the frozen forecasting
pipeline can generalize to later unseen observations. They do not
replace the protected 2025 holdout evaluation and should not be used
retrospectively to select or tune the final model.

## 12. Peak-Risk decision threshold

The official frozen operational threshold remains:

``` text
0.06
```

The XGBoost model produces a continuous `peak_risk_score`; the threshold
converts that score into the binary `peak_alert`.

Changing the threshold does **not** require retraining XGBoost because
it does not change the fitted classifier or its scores. It changes only
the operational decision rule.

Nevertheless, `0.06` should remain the official Capstone threshold
because it was selected during the model-development/refinement process.
Alternative thresholds may later be evaluated as decision-policy
sensitivity scenarios. Any replacement official threshold should be
revalidated and documented using Precision, Recall, F1, false positives,
false negatives, and the operational cost of missed Peaks.

## 13. Runtime and deployment considerations

Performance profiling in IFB_09 showed that feature preparation itself
is fast. The principal deployment bottleneck is the final model
architecture:

``` text
24 Random Forest artifacts
+
24 XGBoost artifacts
```

The Random Forest artifacts are particularly large, so cold model
loading and inference can be expensive in memory and execution time.

The operational implementation therefore supports:

-   model caching within the Python process;
-   separation of cold model loading from inference timing;
-   configurable model-loading workers;
-   loading only the recent history required for a forecast origin.

For memory-constrained local execution, conservative model loading is
preferable.

A future production optimization could investigate smaller Random
Forests, controlled tree depth, model compression, or an alternative
multi-output architecture. Such changes would constitute a new model
version and would require validation. They are not required to complete
the current validated Capstone pipeline.

## 14. Outputs and development artifacts

During development and validation, the IFB can save:

-   model-ready feature CSV/Parquet;
-   operational file inventory;
-   demand/weather coverage reports;
-   feature-contract reports;
-   domain/OOD reports;
-   integrated predictions;
-   prediction-validation reports;
-   runtime profile;
-   post-holdout actual-vs-forecast results;
-   post-holdout metrics;
-   run manifest.

These files provide reproducibility, debugging evidence, and
documentation for the Capstone.

A future Streamlit application does **not** need to write all of these
files for every user prediction. In deployment, the prediction DataFrame
can remain in memory and feed charts, KPIs, alerts, and tables directly.
Export to CSV/Parquet can be optional.

## 15. IFB phase completion status

The Operational Inference Feature Builder phase is considered
**complete** for the current Model v1 scope.

Completed evidence includes:

``` text
Input contract                         PASS
Incremental operational loading       PASS
Recent-demand validation              PASS
Weather-grid validation               PASS
Forecast-grid generation              PASS
Frozen feature-contract validation    PASS
Model-specific rolling compatibility  PASS
Out-of-domain checks                  PASS
Historical replay                     PASS
Integrated RF + XGB inference         PASS
2026 operational simulation           PASS
Final prediction validation           PASS
Post-holdout forecast evaluation      COMPLETED
Runtime profiling                     COMPLETED
```

The IFB therefore provides a validated bridge between operational inputs
and the frozen forecasting/Peak-Risk models.

## 16. Scope limitations carried forward

The following are intentional limitations rather than unresolved IFB
errors:

1.  Model v1 uses forecast-origin weather rather than future target-hour
    weather.
2.  The direct 24-horizon Random Forest architecture produces large
    serialized artifacts and a high cold-start memory/loading cost.
3.  The official Peak-Risk threshold prioritizes high recall and may
    generate a relatively large number of alerts.
4.  Operational inference requires sufficiently recent observed demand;
    the system intentionally refuses to fabricate long missing
    histories.
5.  Out-of-domain weather can be processed, but predictions outside the
    historical development domain should be presented with a reliability
    warning.
6.  The 2026 post-holdout evaluation is supplementary and does not
    replace the protected 2025 holdout.
