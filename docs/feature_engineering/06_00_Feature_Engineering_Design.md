# Feature Engineering Design

## Objective

The Feature Engineering Phase transforms the cleaned analytical dataset into reproducible predictor
variables for the Forecasting and Peak-Risk models while preserving temporal
ordering and preventing data leakage.

## Shared feature families

Both modeling tasks may use:

- existing calendar and FSA variables;
- cyclical temporal encodings;
- current/future weather inputs available at prediction time;
- historical electricity-demand lags;
- leakage-safe rolling demand summaries;
- weather change and rolling summaries;
- a small set of EDA-justified interaction features.

## Forecasting target

`total_consumption_kwh` remains the observed target. No future forecast-horizon targets are
created during this phase.

## Historical target features

Configured lags: `[1, 2, 3, 24, 48, 168]`

Configured rolling windows: `[3, 6, 24, 168]`

All target rolling features are calculated after a one-hour shift, so the
current target value is excluded.

## Peak-Risk design

The final Peak-Risk label is intentionally NOT generated in the static Feature
Engineering dataset. Seasonal/FSA thresholds must be estimated using training
data only inside each temporal validation window.

| candidate_feature         | status                      | reason                                                                  |
|:--------------------------|:----------------------------|:------------------------------------------------------------------------|
| previous_day_peak         | deferred_to_training_window | Requires a Peak threshold fitted only on training data.                 |
| previous_week_peak        | deferred_to_training_window | Requires a Peak threshold fitted only on training data.                 |
| hours_since_previous_peak | deferred_to_training_window | Requires historical Peak labels derived from a training-only threshold. |

## Forecast-Horizon Availability

Historical demand features are valid only when the information required to calculate them 
is available at the time each forecast horizon is generated.

For a 24-hour forecast, the availability of short lags depends on the forecasting strategy. 
For example, `lag_1h` is directly available for the first forecasted hour, but future horizons 
may require either recursively predicted demand values or a direct/multi-output design that 
uses only information available at forecast origin.

Therefore, historical lag and rolling features created during this phase are candidate predictors. 
Their actual availability and construction must be validated separately for each forecasting strategy 
and forecast horizon during the modeling phase.

## Leakage policy

Features are classified into three calculation scopes:

- `static_dataset`: deterministic from timestamp, geography, or existing data;
- `current_exogenous`: may use weather information assumed available from the
  operational weather forecast;
- `historical_only`: must only use observations earlier than the prediction time.

No future observed electricity-demand value is used as a predictor.
