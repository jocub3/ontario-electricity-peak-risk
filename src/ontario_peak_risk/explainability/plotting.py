"""Figures for explainability and model interpretation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def top_importance_plot(
    summary: pd.DataFrame,
    task: str,
    *,
    top_n: int = 15,
):
    frame = (
        summary.loc[summary["task"].eq(task)]
        .nlargest(int(top_n), "mean_importance_pct")
        .sort_values("mean_importance_pct")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(frame["feature"], frame["mean_importance_pct"])
    ax.set_xlabel("Mean built-in importance across horizons (%)")
    ax.set_title(f"Global Feature Importance — {task.upper()}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def family_importance_plot(
    importance: pd.DataFrame,
    task: str,
):
    frame = (
        importance.loc[importance["task"].eq(task)]
        .groupby(["horizon", "feature_family"], observed=True)["importance_pct"]
        .sum()
        .reset_index()
    )
    pivot = frame.pivot(
        index="horizon",
        columns="feature_family",
        values="importance_pct",
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    for family in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[family],
            marker="o",
            label=family,
        )
    ax.set_xticks(range(1, 25))
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Built-in importance (%)")
    ax.set_title(f"Feature-Family Importance by Horizon — {task.upper()}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def importance_heatmap(
    importance: pd.DataFrame,
    task: str,
    *,
    top_n: int = 12,
):
    subset = importance.loc[importance["task"].eq(task)].copy()
    top = (
        subset.groupby("feature", observed=True)["importance_pct"]
        .mean()
        .nlargest(int(top_n))
        .index
    )
    matrix = (
        subset.loc[subset["feature"].isin(top)]
        .pivot(index="feature", columns="horizon", values="importance_pct")
        .fillna(0)
        .reindex(top)
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(matrix.values, aspect="auto")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_xlabel("Forecast horizon")
    ax.set_title(f"Feature Importance Across Horizons — {task.upper()}")
    fig.colorbar(image, ax=ax, label="Importance (%)")
    fig.tight_layout()
    return fig


def shap_global_bar(
    shap_global: pd.DataFrame,
    task: str,
    *,
    top_n: int = 15,
):
    frame = (
        shap_global.loc[shap_global["task"].eq(task)]
        .groupby(["feature", "feature_family"], observed=True)["mean_abs_shap"]
        .mean()
        .reset_index()
        .nlargest(int(top_n), "mean_abs_shap")
        .sort_values("mean_abs_shap")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(frame["feature"], frame["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Global SHAP Importance — {task.upper()}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def weather_effect_plot(
    binned: pd.DataFrame,
    feature: str,
    task: str,
):
    frame = binned.loc[
        binned["task"].eq(task) & binned["feature"].eq(feature)
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    for horizon, group in frame.groupby("horizon", observed=True):
        ordered = group.sort_values("mean_feature_value")
        ax.plot(
            ordered["mean_feature_value"],
            ordered["mean_shap"],
            marker="o",
            label=f"h+{horizon}",
        )
    ax.axhline(0, linestyle="--")
    ax.set_xlabel(feature)
    ax.set_ylabel("Mean SHAP contribution")
    ax.set_title(f"Weather Influence — {feature} — {task.upper()}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def local_contribution_plot(
    local_table: pd.DataFrame,
    *,
    title: str,
):
    frame = local_table.sort_values("shap_contribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(frame["feature"], frame["shap_contribution"])
    ax.axvline(0)
    ax.set_xlabel("SHAP contribution")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def shap_horizon_heatmap(
    shap_global: pd.DataFrame,
    task: str,
    *,
    top_n: int = 12,
):
    subset = shap_global.loc[shap_global["task"].eq(task)].copy()
    top = (
        subset.groupby("feature", observed=True)["mean_abs_shap"]
        .mean()
        .nlargest(int(top_n))
        .index
    )
    matrix = (
        subset.loc[subset["feature"].isin(top)]
        .pivot(index="feature", columns="horizon", values="mean_abs_shap")
        .fillna(0)
        .reindex(top)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(matrix.values, aspect="auto")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"h+{h}" for h in matrix.columns])
    ax.set_xlabel("Representative horizon")
    ax.set_title(f"SHAP Importance Across Horizons — {task.upper()}")
    fig.colorbar(image, ax=ax, label="Mean |SHAP|")
    fig.tight_layout()
    return fig
