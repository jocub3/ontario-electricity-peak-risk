# Target Definition Design

## Objective

Target Definition Phase formally defines the two outcomes that will be predicted in the modeling stages: 24-hour electricity demand and hourly Peak-Risk.

No predictive model is trained in this phase.

## Unit of prediction

The analytical unit is one FSA at one hourly forecast origin.

For each forecast origin `t`, the Forecasting task predicts electricity demand for:

`t+1, t+2, ..., t+24`.

## Forecasting target

The observed source target is:

`total_consumption_kwh`

The forecasting target is represented as a 24-horizon target matrix:

`target_h01 ... target_h24`

Each target value is aligned by explicit `FSA + future timestamp`.

## Peak-Risk target

Peak-Risk is binary:

- `1` = observed electricity demand exceeds the applicable Peak threshold;
- `0` = observed electricity demand does not exceed the applicable Peak threshold.

The configured exploratory/modeling threshold percentile is:

`0.975`

Thresholds are estimated separately by:

`fsa + season`

## Anti-leakage rule

Peak thresholds must never be fitted using future validation or test observations.

The final model must:

1. fit thresholds using the training window only;
2. preserve those fitted thresholds;
3. apply them to validation/test observations without recalculating them from those observations.

## Diagnostic Peak labels

Target Definition Phase may generate walk-forward diagnostic Peak labels for validation and distribution analysis.

For evaluation year `Y`, thresholds are fitted using years strictly earlier than `Y`.

These diagnostic labels are not the final modeling labels and must not replace fold-specific target creation during model evaluation.

## Relationship between Forecasting and Peak-Risk

Both tasks use the same hourly prediction horizon.

Forecasting estimates future electricity consumption, while Peak-Risk estimates whether each future hour represents unusually high demand relative to a training-derived FSA/season threshold.

The two targets are related but are evaluated as separate predictive tasks.
