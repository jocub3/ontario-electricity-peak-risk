"""Documentation generation for Target Definition Phase."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import dataframe_to_markdown


def write_target_design_document(
    docs_dir: Path,
    config: dict,
) -> None:
    """Write the formal target design."""
    settings = config["target_definition"]
    forecasting = settings["forecasting"]
    peak = settings["peak_risk"]

    content = f"""# Target Definition Design

## Objective

Target Definition Phase formally defines the two outcomes that will be predicted in the modeling stages: 24-hour electricity demand and hourly Peak-Risk.

No predictive model is trained in this phase.

## Unit of prediction

The analytical unit is one FSA at one hourly forecast origin.

For each forecast origin `t`, the Forecasting task predicts electricity demand for:

`t+1, t+2, ..., t+{forecasting["horizon_hours"]}`.

## Forecasting target

The observed source target is:

`{forecasting["target_column"]}`

The forecasting target is represented as a 24-horizon target matrix:

`target_h01 ... target_h24`

Each target value is aligned by explicit `FSA + future timestamp`.

## Peak-Risk target

Peak-Risk is binary:

- `1` = observed electricity demand exceeds the applicable Peak threshold;
- `0` = observed electricity demand does not exceed the applicable Peak threshold.

The configured exploratory/modeling threshold percentile is:

`{peak["percentile_threshold"]}`

Thresholds are estimated separately by:

`{" + ".join(peak["grouping_columns"])}`

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
"""

    (
        docs_dir / "07_00_Target_Definition_Design.md"
    ).write_text(
        content,
        encoding="utf-8",
    )


def write_target_summary_document(
    docs_dir: Path,
    forecasting_summary: pd.DataFrame,
    peak_overall: pd.DataFrame,
    forecasting_validation: pd.DataFrame,
    peak_validation: pd.DataFrame,
) -> None:
    """Write the Target Definition Phase completion summary."""
    content = f"""# Target Definition Summary

## Forecasting target

The Forecasting task uses a 24-hour target matrix (`target_h01` through `target_h24`) aligned by FSA and hourly forecast origin.

### Horizon coverage

{dataframe_to_markdown(forecasting_summary)}

## Peak-Risk target

Peak-Risk thresholds are training-derived and are not calculated globally from the full dataset.

Walk-forward diagnostic labels are used only to confirm target behavior without future-year leakage.

### Diagnostic Peak distribution

{dataframe_to_markdown(peak_overall)}

## Forecasting validation

{dataframe_to_markdown(forecasting_validation)}

## Peak-Risk validation

{dataframe_to_markdown(peak_validation)}

## Modeling handoff

The next modeling stages must preserve the following rules:

1. all comparisons must use the same 24-hour target definition;
2. chronological train/validation/test windows must be used;
3. Peak thresholds must be fitted independently inside each training window;
4. validation/test Peak labels must use thresholds learned from training only;
5. forecast-horizon feature availability must be respected;
6. diagnostic Peak labels generated in the Target Definition Phase must not be used as globally precomputed final model labels.
"""

    (
        docs_dir / "07_06_Target_Definition_Summary.md"
    ).write_text(
        content,
        encoding="utf-8",
    )
