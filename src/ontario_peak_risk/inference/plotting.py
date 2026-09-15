from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def demand_heatmap(predictions: pd.DataFrame):
    pivot = predictions.pivot(
        index="fsa",
        columns="horizon",
        values="forecast_consumption_kwh",
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    image = ax.imshow(pivot.values, aspect="auto")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("FSA")
    ax.set_title("24-Hour Forecast Demand Heatmap")

    fig.colorbar(
        image,
        ax=ax,
        label="Forecast consumption (kWh)",
    )

    fig.tight_layout()
    return fig


def peak_risk_heatmap(predictions: pd.DataFrame):
    pivot = predictions.pivot(
        index="fsa",
        columns="horizon",
        values="peak_risk_score",
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    image = ax.imshow(pivot.values, aspect="auto")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("FSA")
    ax.set_title("24-Hour Peak-Risk Score Heatmap")

    fig.colorbar(
        image,
        ax=ax,
        label="Peak-Risk score",
    )

    fig.tight_layout()
    return fig


def peak_alert_heatmap(predictions: pd.DataFrame):
    pivot = (
        predictions
        .assign(peak_alert_int=predictions["peak_alert"].astype(int))
        .pivot(
            index="fsa",
            columns="horizon",
            values="peak_alert_int",
        )
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    image = ax.imshow(
        pivot.values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("FSA")
    ax.set_title("24-Hour Peak Alert Heatmap (1 = Alert)")

    fig.colorbar(
        image,
        ax=ax,
        ticks=[0, 1],
        label="Peak Alert",
    )

    fig.tight_layout()
    return fig


def demand_lines_with_peak_alerts(predictions: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 6))

    for fsa, group in predictions.groupby("fsa", observed=True):
        group = group.sort_values("horizon")

        ax.plot(
            group["horizon"],
            group["forecast_consumption_kwh"],
            marker="o",
            label=fsa,
        )

        alerts = group.loc[group["peak_alert"]]

        if not alerts.empty:
            ax.scatter(
                alerts["horizon"],
                alerts["forecast_consumption_kwh"],
                s=80,
            )

    ax.set_xticks(range(1, 25))
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Forecast consumption (kWh)")
    ax.set_title("24-Hour Demand Forecast with Peak-Risk Alerts")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def peak_risk_lines(predictions: pd.DataFrame, operational_threshold: float = 0.06):
    fig, ax = plt.subplots(figsize=(13, 6))

    for fsa, group in predictions.groupby("fsa", observed=True):
        group = group.sort_values("horizon")

        ax.plot(
            group["horizon"],
            group["peak_risk_score"],
            marker="o",
            label=fsa,
        )

    ax.axhline(
        operational_threshold,
        linestyle="--",
        label=f"Operational threshold = {operational_threshold:.2f}",
    )

    ax.set_xticks(range(1, 25))
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Peak-Risk score")
    ax.set_title("24-Hour Peak-Risk Score by FSA")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def max_demand_by_fsa(predictions: pd.DataFrame):
    summary = (
        predictions
        .groupby("fsa", observed=True)
        .agg(
            max_forecast_consumption_kwh=("forecast_consumption_kwh", "max"),
            peak_alert_hours=("peak_alert", "sum"),
            max_peak_risk_score=("peak_risk_score", "max"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        summary["fsa"],
        summary["max_forecast_consumption_kwh"],
    )
    ax.set_xlabel("FSA")
    ax.set_ylabel("Maximum forecast consumption (kWh)")
    ax.set_title("Maximum 24-Hour Forecast Demand by FSA")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    return fig


def alert_count_by_fsa(predictions: pd.DataFrame):
    summary = (
        predictions
        .groupby("fsa", observed=True)["peak_alert"]
        .sum()
        .reset_index(name="peak_alert_hours")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        summary["fsa"],
        summary["peak_alert_hours"],
    )
    ax.set_xlabel("FSA")
    ax.set_ylabel("Peak alert hours")
    ax.set_title("Peak-Risk Alert Count by FSA")
    ax.set_ylim(0, 24)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    return fig
