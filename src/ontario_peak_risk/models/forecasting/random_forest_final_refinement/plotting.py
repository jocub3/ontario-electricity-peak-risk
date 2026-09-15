"""Visual diagnostics for the controlled Random Forest refinement."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def mean_mae_plot(ranking: pd.DataFrame):
    data = ranking.sort_values("mean_mae")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(data["refinement_id"], data["mean_mae"])
    ax.set_ylabel("Mean MAE (kWh)")
    ax.set_title("Random Forest Final Refinement — Mean MAE")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def fold_mae_plot(details: pd.DataFrame):
    if "fold" not in details.columns:
        return None

    pivot = (
        details
        .groupby(["refinement_id", "fold"], observed=True)["mae"]
        .mean()
        .unstack("fold")
    )

    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Random Forest Refinement — MAE by Fold")
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=25,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    return ax.get_figure()


def horizon_mae_plot(details: pd.DataFrame):
    if "horizon" not in details.columns:
        return None

    grouped = (
        details
        .groupby(["refinement_id", "horizon"], observed=True)["mae"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    for model_id, group in grouped.groupby(
        "refinement_id",
        observed=True,
    ):
        group = group.sort_values("horizon")
        ax.plot(
            group["horizon"],
            group["mae"],
            marker="o",
            label=model_id,
        )

    ax.set_xlabel("Representative Forecast Horizon")
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Random Forest Refinement — MAE by Representative Horizon")
    ax.set_xticks(sorted(grouped["horizon"].unique()))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
