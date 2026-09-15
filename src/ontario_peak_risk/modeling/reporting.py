"""Documentation generation for Modeling Foundation Phase."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import dataframe_to_markdown


def write_modeling_design(
    docs_dir: Path,
    config: dict,
) -> None:
    """Write the shared modeling design document."""

    forecasting = (
        config["modeling"]["forecasting"]
    )

    peak_risk = (
        config["modeling"]["peak_risk"]
    )

    content = f"""# Modeling Foundation Design

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

Because both tasks predict the following {forecasting["horizons"]} hours, a forecast origin is eligible only when its complete target horizon remains inside the same training, validation, or test period.

For example, if a training period ends at `2022-12-31 23:00`, the final valid 24-hour forecast origin is `2022-12-30 23:00`.

This prevents future target observations from crossing a temporal fold boundary.

## Preprocessing Policy

Preprocessing is model-specific but must follow a common anti-leakage rule:

> Any transformation that learns parameters from data must be fitted on the training portion only.

Examples include scaling, imputation, encoding, and learned transformations.

The framework does not globally standardize `feature_dataset.parquet`.

## Forecasting Evaluation

Forecasting models predict:

`target_h01 ... target_h{forecasting["horizons"]:02d}`

Performance must be reported:

- globally;
- by validation fold;
- by FSA when applicable;
- by forecast horizon.

Primary metrics are:

{", ".join(forecasting["primary_metrics"])}

The common model-selection metric is:

`{forecasting["selection_metric"]}`

Other metrics remain mandatory supporting evidence.

## Peak-Risk Evaluation

Peak-Risk models predict the Peak status of each of the following {peak_risk["horizons"]} hours:

`peak_h01 ... peak_h{peak_risk["horizons"]:02d}`

Peak thresholds are defined using the {peak_risk["percentile_threshold"]:.3f} quantile and grouping variables:

`{", ".join(peak_risk["threshold_grouping"])}`

Thresholds must be fitted inside each training window and then applied unchanged to the corresponding validation or test observations.

The initial probability threshold used for fair classifier comparison is:

`{peak_risk["probability_threshold"]:.2f}`

The common Peak-Risk selection metric is:

`{peak_risk["selection_metric"]}`

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
"""

    (
        docs_dir
        / "08_00_Modeling_Foundation_Design.md"
    ).write_text(
        content,
        encoding="utf-8",
    )


def write_modeling_summary(
    docs_dir: Path,
    split_summary: pd.DataFrame,
    metric_dictionary: pd.DataFrame,
    framework_validation: pd.DataFrame,
    fsa_coverage: pd.DataFrame,
    peak_horizon_policy: pd.DataFrame,
) -> None:
    """Write the Modeling Foundation Phase summary."""

    content = f"""# Modeling Foundation Summary

## Common Horizon-Safe Temporal Splits

{dataframe_to_markdown(split_summary)}

## FSA Coverage by Fold

{dataframe_to_markdown(fsa_coverage)}

## Official Project Metrics

{dataframe_to_markdown(metric_dictionary)}

## Peak-Risk Multi-Horizon Policy

{dataframe_to_markdown(peak_horizon_policy)}

## Framework Validation

{dataframe_to_markdown(framework_validation)}

## Handoff to Model Branches

After this framework is reviewed and merged into `develop`, individual model branches should be created from the updated `develop` branch.

The first model branches will be created separately:

- `forecasting/seasonal-naive-baseline`
- `peak-risk/logistic-regression-baseline`

Neither baseline is implemented in the Modeling Foundation Phase's `modeling/framework` branch.

All future model branches must reuse the common:

- temporal folds;
- horizon-safe forecast-origin boundaries;
- target definitions;
- evaluation metrics;
- model-selection metrics;
- Peak-Risk threshold policy;
- classification comparison threshold;
- preprocessing and anti-leakage rules.
"""

    (
        docs_dir
        / "08_09_Modeling_Foundation_Summary.md"
    ).write_text(
        content,
        encoding="utf-8",
    )
