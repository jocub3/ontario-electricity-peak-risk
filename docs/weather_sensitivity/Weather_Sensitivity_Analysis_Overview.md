# Weather Sensitivity Analysis — Phase Overview

## 1. Scope

This phase evaluates the sensitivity of the frozen Model v1 to controlled
perturbations of `origin__Temp (°C)`.

Only the temperature feature is modified. All other model inputs remain fixed,
and neither the Random Forest demand-forecasting models nor the XGBoost
Peak-Risk models are retrained.

The analysis therefore measures **model sensitivity**, not causal weather
effects. It also does not represent target-hour future-weather sensitivity,
because Model v1 uses weather observed at the forecast origin.

## 2. Scenarios

Five controlled temperature scenarios were evaluated:

- −5.0 °C
- −2.5 °C
- Baseline
- +2.5 °C
- +5.0 °C

Each scenario was evaluated across:

- 6 FSAs
- 24 forecast horizons
- Random Forest electricity-demand forecasts
- XGBoost Peak-Risk scores and alert decisions

Total scenario predictions: **720**

## 3. Scenario-Level Results

| Scenario | Δ Temperature (°C) | Mean ABS(Δ Forecast) (kWh) | Max ABS(Δ Forecast) (kWh) | Mean ABS(Δ Peak-Risk) | Max ABS(Δ Peak-Risk) | Alert Changes | Out of Range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -5C | -5.000 | 226.359 | 774.131 | 0.056 | 0.408 | 8 | 0 |
| -2.5C | -2.500 | 89.276 | 536.643 | 0.038 | 0.273 | 6 | 0 |
| Baseline | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| +2.5C | 2.500 | 48.947 | 205.985 | 0.052 | 0.425 | 5 | 0 |
| +5C | 5.000 | 80.608 | 293.257 | 0.106 | 0.596 | 17 | 0 |

## 4. Key Findings

### Demand Forecast Sensitivity

The Random Forest forecasts respond meaningfully to changes in forecast-origin
temperature.

The average absolute forecast response was:

- **89.28 kWh** for −2.5 °C
- **226.36 kWh** for −5.0 °C
- **48.95 kWh** for +2.5 °C
- **80.61 kWh** for +5.0 °C

The response is not perfectly symmetric between warmer and colder scenarios,
indicating nonlinear behavior in the fitted forecasting model.

### Peak-Risk Sensitivity

Temperature perturbations also modify the XGBoost Peak-Risk score.

Across all scenarios, **36 alert decisions** changed relative to
the Baseline scenario using the frozen operational threshold of
**0.06**.

This demonstrates that temperature variation can influence not only the
continuous risk score but, in some FSA/horizon combinations, the final
operational alert decision.

### FSA and Horizon Differences

Sensitivity is heterogeneous across both geography and forecast horizon.
Some FSA/horizon combinations respond substantially more strongly than others.

The largest absolute demand-forecast response observed was
**774.13 kWh**, for:

- FSA: **M6G**
- Horizon: **h+14**
- Scenario: **-5C**

This supports evaluating sensitivity at the FSA/horizon level rather than
relying only on province-wide averages.

### Nonlinearity and Asymmetry

The ±2.5 °C and ±5 °C scenarios do not generate perfectly mirrored model
responses.

This asymmetry is evidence of nonlinear model behavior and interactions among
the frozen model features. It must not be interpreted as evidence of a causal
relationship between temperature and electricity demand or Peak-Risk.

## 5. Domain Validation

Temperature-domain validation produced:

- Total scenario predictions: **720**
- NORMAL: **720**
- CAUTION: **0**
- OUT_OF_RANGE: **0**

All evaluated scenarios therefore remained within the validated temperature
support of Model v1 for this operational forecast origin.

## 6. Interpretation Boundaries

The following limitations apply:

1. The analysis perturbs only `origin__Temp (°C)`.
2. All other model features remain fixed.
3. Model parameters are frozen.
4. Results represent model response, not causal effects.
5. The analysis does not simulate complete alternative weather trajectories.
6. Target-hour future weather is not currently part of Model v1.
7. The official Peak-Risk decision threshold remains fixed at
   **0.06**.

A future Model v2 could incorporate target-hour weather forecasts and permit
full future-weather scenario analysis.

## 7. Decision-Support Relevance

The sensitivity analysis provides an additional decision-support layer beyond
the baseline forecast.

The final application can therefore present:

- baseline 24-hour demand forecasts;
- baseline Peak-Risk probabilities and alerts;
- controlled temperature scenarios;
- changes in expected electricity demand;
- changes in Peak-Risk score;
- alert transitions relative to baseline;
- warnings when scenarios approach or exceed the model-development domain.

This functionality should be interpreted as **scenario-based model
sensitivity**, rather than deterministic prediction under alternative weather
conditions.

## 8. Main Artifacts

- `WSA_sensitivity_master.parquet`
- `WSA_sensitivity_master.csv`
- `WSA_02_rf_forecast_sensitivity.csv`
- `WSA_03_xgb_peak_risk_sensitivity.csv`
- `WSA_03_alert_transitions.csv`
- `WSA_04_fsa_scenario_summary.csv`
- `WSA_04_horizon_summary.csv`
- `WSA_04_symmetry_analysis.csv`
- `WSA_06_validation_summary.csv`
- `WSA_06_nonlinearity_symmetry.csv`
- comparative figures under `reports/figures/weather_sensitivity/`

## 9. Phase Status

**Weather Sensitivity Analysis: COMPLETE**

The outputs are ready for integration into the final Decision-Support
Application.
