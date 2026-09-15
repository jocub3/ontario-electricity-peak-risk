"""Local explanation helpers for individual RF and XGBoost predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prediction_value(
    pipeline,
    X: pd.DataFrame,
    task: str,
) -> float:
    if len(X) != 1:
        raise ValueError("Local explanation requires exactly one row.")

    if task == "rf":
        return float(pipeline.predict(X)[0])
    if task == "xgb":
        return float(pipeline.predict_proba(X)[0, 1])
    raise ValueError("task must be 'rf' or 'xgb'.")


def local_contribution_table(
    X: pd.DataFrame,
    shap_result: dict,
    *,
    task: str,
    horizon: int,
    top_n: int = 10,
) -> pd.DataFrame:
    if len(X) != 1:
        raise ValueError("Local contribution table requires one row.")

    names = shap_result["public_feature_names"]
    values = shap_result["shap_values_public"][0]

    rows = []
    for index, feature in enumerate(names):
        raw_value = X.iloc[0][feature] if feature in X.columns else pd.NA
        contribution = float(values[index])

        if task == "xgb":
            direction = (
                "increases risk score"
                if contribution > 0
                else "decreases risk score"
                if contribution < 0
                else "neutral"
            )
        else:
            direction = (
                "increases forecast"
                if contribution > 0
                else "decreases forecast"
                if contribution < 0
                else "neutral"
            )

        rows.append(
            {
                "task": task,
                "horizon": int(horizon),
                "feature": feature,
                "feature_value": raw_value,
                "shap_contribution": contribution,
                "abs_contribution": abs(contribution),
                "direction": direction,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_contribution", ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )


def human_readable_local_summary(
    table: pd.DataFrame,
    *,
    prediction: float,
    task: str,
    fsa: str,
    target_timestamp,
) -> str:
    target = pd.Timestamp(target_timestamp)

    if task == "rf":
        heading = (
            f"Demand forecast explanation — {fsa}, "
            f"{target:%Y-%m-%d %H:%M}: {prediction:,.1f} kWh"
        )
    else:
        heading = (
            f"Peak-Risk explanation — {fsa}, "
            f"{target:%Y-%m-%d %H:%M}: score={prediction:.3f}"
        )

    lines = [heading, ""]
    for _, row in table.iterrows():
        lines.append(
            f"- {row['feature']} = {row['feature_value']}: "
            f"{row['direction']} "
            f"(SHAP={row['shap_contribution']:+.4f})"
        )

    if task == "xgb":
        lines.extend(
            [
                "",
                "Note: tree SHAP contributions are interpreted on the model's "
                "native/raw output scale unless probability-output SHAP is "
                "explicitly configured. The displayed Peak-Risk probability "
                "is reported separately.",
            ]
        )
    return "\n".join(lines)
