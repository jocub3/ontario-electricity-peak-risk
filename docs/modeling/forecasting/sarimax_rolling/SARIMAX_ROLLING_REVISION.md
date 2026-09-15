# SARIMAX Rolling Revision — Before vs After

This revision intentionally **does not replace** the original SARIMAX files.

## Original strategy

The original implementation:

- completed moderate tuning on 3 specifications;
- selected `(2,0,1) × (1,0,0,24)` with constant trend;
- then fitted one SARIMAX for every:
  - validation fold;
  - FSA;
  - forecast horizon.

For a complete evaluation this implies approximately:

`2 folds × 6 FSAs × 24 horizons = 288 SARIMAX fits`

after the tuning stage.

## Revised strategy

The revised implementation:

- reuses the original selected specification;
- fits **one SARIMAX per FSA and fold**;
- updates the fitted state when a new hourly observation becomes available;
- generates a rolling 24-hour forecast at every validation origin;
- therefore requires approximately:

`2 folds × 6 FSAs = 12 SARIMAX fits`

The 24 forecast horizons are produced from each fitted model rather than
fitting 24 separate models.

## Why future observed weather is not used

A single rolling SARIMAX requires exogenous variables for every future
timestamp in the 24-hour forecast.

Observed future weather would not be known at prediction time. The revised
implementation therefore uses deterministic target-time calendar regressors
(hour, weekday, month, weekend) and excludes future observed weather.

A future operational version could use an actual weather-forecast feed.

## Runtime controls

The revised runner adds:

- smoke-test mode;
- per-fit convergence status;
- fit runtime;
- optimizer iteration count;
- progress every N forecast origins;
- task-level ETA;
- estimated remaining total runtime;
- checkpoint files by fold/FSA;
- resume support.

## Recommended execution order

1. Run smoke test.
2. Confirm convergence and runtime.
3. If acceptable, run the full version.
4. Compare original tuning outputs with revised validation outputs.
5. Do not delete the original SARIMAX implementation.
