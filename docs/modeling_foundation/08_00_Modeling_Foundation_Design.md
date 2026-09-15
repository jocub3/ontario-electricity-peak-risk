# Modeling Foundation Design

## Objective

The Modeling Foundation Phase establishes the common modeling rules that every Forecasting and Peak-Risk model must follow.

No baseline or candidate predictive model is implemented in the `modeling/framework` branch.

## Core Principle

All models must be compared under the same experimental conditions.

The shared framework defines:

- chronological training, validation, and final test periods;
- horizon-safe forecast-origin boundaries;
- common target definitions;
- common evaluation metrics;
- preprocessing rules;
- Peak-Risk threshold rules;
- model-selection rules;
- reporting requirements;
- anti-leakage safeguards.

## Temporal Validation Strategy

Model development uses expanding-window validation.

The final 2025 holdout is reserved for final evaluation and must not be used for model selection, preprocessing decisions, classification-threshold optimization, or hyperparameter tuning.

### Horizon-safe split rule

The configured split dates represent observation-period boundaries.

Because both tasks predict the following 24 hours, a forecast origin is eligible only when its complete target horizon remains inside the same training, validation, or test period.

For example, if a training period ends at `2022-12-31 23:00`, the final valid 24-hour forecast origin is `2022-12-30 23:00`.

This prevents future target observations from crossing a temporal fold boundary.

## Preprocessing Policy

Preprocessing is model-specific but must follow a common anti-leakage rule:

> Any transformation that learns parameters from data must be fitted on the training portion only.

Examples include scaling, imputation, encoding, and learned transformations.

The framework does not globally standardize `feature_dataset.parquet`.

## Forecasting Evaluation

Forecasting models predict:

`target_h01 ... target_h24`

Performance must be reported:

- globally;
- by validation fold;
- by FSA when applicable;
- by forecast horizon.

Primary metrics are:

mae, rmse, mape

The common model-selection metric is:

`mae`

Other metrics remain mandatory supporting evidence.

## Peak-Risk Evaluation

Peak-Risk models predict the Peak status of each of the following 24 hours:

`peak_h01 ... peak_h24`

Peak thresholds are defined using the 0.975 quantile and grouping variables:

`fsa, season`

Thresholds must be fitted inside each training window and then applied unchanged to the corresponding validation or test observations.

The initial probability threshold used for fair classifier comparison is:

`0.50`

The common Peak-Risk selection metric is:

`pr_auc`

Alternative operational probability thresholds may be investigated later using validation data only.

## Model-Branch Responsibility

Each future model branch may implement its own model-specific preprocessing, fitting logic, hyperparameters, and forecasting strategy.

However, model branches must not redefine:

- temporal folds;
- horizon-safe split rules;
- target definitions;
- official metrics;
- Peak threshold methodology;
- initial classification probability threshold;
- common model-selection metrics;

without an approved update to the Modeling Foundation.
