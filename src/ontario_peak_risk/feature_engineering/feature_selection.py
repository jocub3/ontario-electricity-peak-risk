"""Preliminary feature classification and redundancy review."""

from __future__ import annotations

import numpy as np
import pandas as pd


def preliminary_feature_selection(
    frame: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Classify features for later model-specific selection.

    No feature is automatically removed based solely on this report.
    """
    settings = config["feature_engineering"]["selection"]
    target = config["feature_engineering"]["target_column"]
    traceability = set(config["feature_engineering"]["traceability_columns"])

    dictionary = feature_dictionary.drop_duplicates(
        subset=["feature_name"],
        keep="last",
    ).set_index("feature_name")

    rows = []

    for column in frame.columns:
        missing_pct = float(frame[column].isna().mean() * 100)
        unique_count = int(frame[column].nunique(dropna=True))

        family = (
            dictionary.loc[column, "feature_family"]
            if column in dictionary.index
            else "unknown"
        )

        if column == target:
            priority = "target"
            rationale = "Observed target; excluded from predictor set."
        elif column in traceability:
            priority = "administrative"
            rationale = (
                "Retain for alignment or audit, not as a direct predictor."
            )
        elif column == "reported_premise_count":
            priority = "medium"
            rationale = (
                "Potentially informative operational variable, but forecast-time "
                "availability must be confirmed before inclusion in the final "
                "predictor set."
            )
        elif unique_count <= 1:
            priority = "redundant"
            rationale = "No variance."
        elif missing_pct >= settings["high_missingness_threshold_pct"]:
            priority = "low"
            rationale = (
                "Very high missingness; model-specific handling required."
            )
        elif missing_pct >= settings["medium_missingness_threshold_pct"]:
            priority = "medium"
            rationale = (
                "Substantial missingness; usefulness must justify treatment."
            )
        elif family in {
            "target_lag",
            "target_rolling",
            "temporal_cyclical",
            "weather_nonlinear",
            "weather_change",
            "weather_rolling",
            "spatial",
        }:
            priority = "high"
            rationale = "EDA-supported feature family."
        else:
            priority = "medium"
            rationale = "Retain for model-specific evaluation."

        rows.append(
            {
                "feature": column,
                "feature_family": family,
                "dtype": str(frame[column].dtype),
                "missing_pct": round(missing_pct, 6),
                "unique_count": unique_count,
                "preliminary_priority": priority,
                "rationale": rationale,
            }
        )

    return pd.DataFrame(rows)


def high_correlation_pairs(
    frame: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Identify highly correlated numeric feature pairs."""
    threshold = float(
        config["feature_engineering"]["selection"]["correlation_redundancy_threshold"]
    )

    numeric = frame.select_dtypes(include=[np.number]).copy()
    target = config["feature_engineering"]["target_column"]

    if target in numeric.columns:
        numeric = numeric.drop(columns=[target])

    correlation = numeric.corr(method="pearson")

    rows = []
    columns = correlation.columns.tolist()

    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "pearson_correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values("absolute_correlation", ascending=False)
        .reset_index(drop=True)
        if rows
        else pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "pearson_correlation",
                "absolute_correlation",
            ]
        )
    )
