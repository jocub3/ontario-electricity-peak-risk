"""Markdown documentation writers for the Explainability phase."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame is None or frame.empty:
        return "_No rows available._"
    return frame.head(int(max_rows)).to_markdown(index=False)


def write_global_importance_report(
    path: Path,
    global_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> None:
    content = f"""# EXPL_01 — Global Feature Importance

## Purpose

This report documents built-in tree feature importance from the frozen final
Random Forest forecasting and XGBoost Peak-Risk artifacts. One-hot encoded
categorical variables are aggregated back to their public feature names.

## Global feature ranking

{_md_table(global_summary, 40)}

## Feature-family summary

{_md_table(family_summary, 30)}

## Interpretation policy

Built-in tree importance describes how much fitted trees use each transformed
predictor to reduce their training objective. It is not causal and should not
be interpreted as the direction of effect. SHAP analysis is used in the next
stage to add directional and local interpretation.
"""
    path.write_text(content, encoding="utf-8")


def write_shap_report(
    path: Path,
    shap_global: pd.DataFrame,
) -> None:
    summary = (
        shap_global.groupby(
            ["task", "feature", "feature_family"],
            observed=True,
        )["mean_abs_shap"]
        .mean()
        .reset_index()
        .sort_values(["task", "mean_abs_shap"], ascending=[True, False])
    )
    content = f"""# EXPL_02 — Global SHAP Analysis

## Purpose

SHAP was calculated using a reproducible sample and representative forecast
horizons to limit memory pressure from the large horizon-specific Random Forest
artifacts.

## Mean absolute SHAP ranking

{_md_table(summary, 40)}

## Scope

SHAP values explain the fitted final models; they do not establish causality.
For the binary XGBoost classifier, local SHAP contributions are interpreted on
the model's native/raw output scale unless explicitly configured otherwise.
The operational Peak-Risk score is reported separately.
"""
    path.write_text(content, encoding="utf-8")


def write_weather_report(
    path: Path,
    weather_summary: pd.DataFrame,
    weather_binned: pd.DataFrame,
) -> None:
    content = f"""# EXPL_03 — Weather Influence

## Purpose

This section isolates the contribution of forecast-origin temperature,
relative humidity, and station pressure in the frozen Model v1.

This is **model interpretation**, not weather scenario analysis. Model v1 uses
weather at the forecast origin; it does not use target-hour future weather as
a frozen predictor.

## Weather share of SHAP importance

{_md_table(weather_summary, 30)}

## Binned weather effects

{_md_table(weather_binned, 50)}

## Interpretation constraint

Observed SHAP patterns describe how the fitted models use weather jointly with
demand-history, calendar, FSA, and contextual variables. They are not causal
effects and should not be described as the result of changing weather while
holding every other process constant. Controlled perturbation belongs to the
later Weather Sensitivity phase.
"""
    path.write_text(content, encoding="utf-8")


def write_local_report(
    path: Path,
    rf_table: pd.DataFrame,
    xgb_table: pd.DataFrame,
    summary_text: str,
) -> None:
    content = f"""# EXPL_04 / EXPL_06 — Local Prediction Explanation

{summary_text}

## Forecasting contribution table

{_md_table(rf_table, 20)}

## Peak-Risk contribution table

{_md_table(xgb_table, 20)}

## Interpretation note

The explanation is local to the selected FSA, forecast origin, and target
horizon. A feature contribution is not a causal effect. For XGBoost, the
operational alert threshold is a separate decision rule applied after the model
produces its risk score.
"""
    path.write_text(content, encoding="utf-8")


def write_phase_overview(
    path: Path,
    *,
    completed_sections: list[str],
    key_findings: list[str],
) -> None:
    findings = "\n".join(f"- {item}" for item in key_findings)
    sections = "\n".join(f"- {item}" for item in completed_sections)
    content = f"""# Explainability & Model Interpretation — Phase Overview

## Completed sections

{sections}

## Key findings

{findings}

## Methodological boundaries

- Explainability describes fitted-model behavior and is not causal inference.
- Model v1 weather interpretation applies to forecast-origin weather.
- SHAP sampling is intentionally memory-conscious because final Random Forest
  artifacts are large.
- The official Peak-Risk threshold remains a decision-policy layer separate
  from the XGBoost explanation.
- Weather perturbation scenarios are handled in the subsequent Weather
  Sensitivity / Scenario Analysis phase.
"""
    path.write_text(content, encoding="utf-8")
