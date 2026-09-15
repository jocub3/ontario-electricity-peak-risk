"""Summaries of weather influence derived from SHAP, not scenario perturbations."""

from __future__ import annotations

import pandas as pd


def weather_shap_summary(
    shap_global: pd.DataFrame,
    weather_features: list[str],
) -> pd.DataFrame:
    weather = shap_global.loc[
        shap_global["feature"].isin(weather_features)
    ].copy()

    total = (
        shap_global.groupby(["task", "horizon"], observed=True)["mean_abs_shap"]
        .sum()
        .rename("total_mean_abs_shap")
        .reset_index()
    )

    weather_total = (
        weather.groupby(["task", "horizon"], observed=True)["mean_abs_shap"]
        .sum()
        .rename("weather_mean_abs_shap")
        .reset_index()
    )

    summary = total.merge(
        weather_total,
        on=["task", "horizon"],
        how="left",
    )
    summary["weather_mean_abs_shap"] = (
        summary["weather_mean_abs_shap"].fillna(0.0)
    )
    summary["weather_share_pct"] = (
        summary["weather_mean_abs_shap"]
        / summary["total_mean_abs_shap"]
        * 100
    )
    return summary


def binned_weather_effect(
    long_frame: pd.DataFrame,
    feature: str,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    frame = long_frame.loc[
        long_frame["feature"].eq(feature)
    ].copy()

    numeric = pd.to_numeric(frame["feature_value"], errors="coerce")
    frame = frame.loc[numeric.notna()].copy()
    frame["numeric_value"] = numeric.loc[numeric.notna()]

    if frame.empty:
        return pd.DataFrame()

    unique = frame["numeric_value"].nunique()
    q = min(int(bins), int(unique))
    if q < 2:
        return pd.DataFrame()

    frame["value_bin"] = pd.qcut(
        frame["numeric_value"],
        q=q,
        duplicates="drop",
    )

    return (
        frame.groupby(
            ["task", "horizon", "feature", "value_bin"],
            observed=True,
        )
        .agg(
            observations=("shap_value", "size"),
            mean_feature_value=("numeric_value", "mean"),
            mean_shap=("shap_value", "mean"),
            mean_abs_shap=("shap_value", lambda s: s.abs().mean()),
        )
        .reset_index()
    )
