"""Create the baseline feature dataset without engineered variables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import make_feature_log


def build_base_dataset(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the baseline modeling table.

    Only explicitly administrative/redundant columns are removed.
    No engineered variables are created in this step.
    """
    output = frame.copy()

    drop_columns = [
        column
        for column in config["feature_engineering"].get("baseline_drop_columns", [])
        if column in output.columns
    ]

    output = output.drop(columns=drop_columns)

    log_rows = []
    traceability = set(
        config["feature_engineering"].get("traceability_columns", [])
    )
    target = config["feature_engineering"]["target_column"]

    for column in output.columns:
        if column == target:
            family = "target"
            forecasting = False
            peak = False
            description = "Observed electricity consumption target."
            scope = "target_only"
        elif column in traceability:
            family = "traceability"
            forecasting = False
            peak = False
            description = (
                "Retained for audit, geographic reference, or alignment, "
                "but excluded from the initial predictor set."
            )
            scope = "static_dataset"
        elif column == "reported_premise_count":
            family = "operational_exogenous"
            forecasting = True
            peak = True
            description = (
                "Number of premises represented in the hourly consumption record. "
                "Potential predictor, but its availability at forecast time must be "
                "confirmed before use in production forecasting."
            )
            scope = "availability_review"

        elif column == "fsa":
            family = "spatial"
            forecasting = True
            peak = True
            description = "Forward Sortation Area identifier."
            scope = "static_dataset"        
        else:
            family = "existing"
            forecasting = True
            peak = True
            description = "Existing cleaned predictor retained for preliminary modeling review."
            scope = "static_dataset"

        log_rows.append(
            make_feature_log(
                column,
                family,
                [column],
                description,
                forecasting,
                peak,
                scope,
                "none" if family != "target" else "target_only",
            )
        )

    for column in drop_columns:
        log_rows.append(
            {
                "feature_name": column,
                "feature_family": "administrative_or_redundant",
                "source_columns": column,
                "description": "Removed from the baseline feature table by explicit configuration.",
                "forecasting_candidate": False,
                "peak_risk_candidate": False,
                "computation_scope": "static_dataset",
                "leakage_risk": "none",
            }
        )

    return output, pd.DataFrame(log_rows)
