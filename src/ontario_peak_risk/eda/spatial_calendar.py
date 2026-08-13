"""Spatial FSA comparisons and calendar-effect analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import save_figure, save_table, write_section_markdown


def spatial_summary(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compare target scale and dispersion across FSAs."""
    return (
        frame.groupby("fsa", observed=True)[target]
        .agg(
            hourly_mean="mean",
            hourly_median="median",
            hourly_standard_deviation="std",
            hourly_minimum="min",
            hourly_maximum="max",
            p95=lambda series: series.quantile(0.95),
            p975=lambda series: series.quantile(0.975),
            p99=lambda series: series.quantile(0.99),
        )
        .reset_index()
    )


def hourly_profile_by_fsa(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    return (
        frame.groupby(["fsa", "hour"], observed=True)[target]
        .agg(mean="mean", median="median", maximum="max")
        .reset_index()
    )


def calendar_effects(frame: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    """Summarize target by key calendar dimensions."""
    dimensions = {
        "weekend": ["is_weekend"],
        "holiday": ["is_public_holiday"],
        "season": ["season"],
        "dst": ["is_daylight_saving_time"],
        "dst_transition": ["is_dst_transition_day"],
        "quarter": ["quarter"],
        "month": ["month", "month_name"],
        "hour": ["hour"],
        "workday": ["is_workday"],
        "long_weekend": ["is_long_weekend"],
    }

    outputs = {}

    for name, grouping in dimensions.items():
        outputs[name] = (
            frame.groupby(grouping, observed=True)[target]
            .agg(
                count="count",
                mean="mean",
                median="median",
                maximum="max",
            )
            .reset_index()
        )

    return outputs


def plot_spatial_summary(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(summary["fsa"].astype(str), summary["hourly_mean"])
    axis.set_xlabel("FSA")
    axis.set_ylabel("Mean hourly consumption (kWh)")
    axis.set_title("Mean hourly electricity consumption by FSA")
    save_figure(figure, output_path)


def plot_hourly_profiles(
    profile: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))

    for fsa, group in profile.groupby("fsa", observed=True):
        axis.plot(group["hour"], group["mean"], marker="o", label=str(fsa))

    axis.set_xlabel("Hour")
    axis.set_ylabel("Mean consumption (kWh)")
    axis.set_title("Hourly demand profiles by FSA")
    axis.legend(ncol=3)
    save_figure(figure, output_path)


def plot_calendar_binary(
    table: pd.DataFrame,
    field: str,
    output_path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(table[field].astype(str), table["mean"])
    axis.set_xlabel(field.replace("_", " ").title())
    axis.set_ylabel("Mean consumption (kWh)")
    axis.set_title(title)
    save_figure(figure, output_path)


def seasonal_profile_by_fsa(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Calculate mean, median, and maximum electricity consumption
    by FSA and meteorological season.
    """
    season_order = [
        "Winter",
        "Spring",
        "Summer",
        "Fall",
    ]

    working = frame.copy()

    working["season"] = pd.Categorical(
        working["season"],
        categories=season_order,
        ordered=True,
    )

    return (
        working.groupby(
            ["fsa", "season"],
            observed=True,
        )[target]
        .agg(
            mean="mean",
            median="median",
            maximum="max",
        )
        .reset_index()
        .sort_values(
            ["fsa", "season"]
        )
    )


def plot_seasonal_profiles_by_fsa(
    profile: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compare seasonal electricity-demand profiles across FSAs.
    """
    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for fsa, group in profile.groupby(
        "fsa",
        observed=True,
    ):
        axis.plot(
            group["season"].astype(str),
            group["mean"],
            marker="o",
            label=str(fsa),
        )

    axis.set_xlabel("Season")
    axis.set_ylabel(
        "Mean consumption (kWh)"
    )
    axis.set_title(
        "Seasonal electricity-demand profiles by FSA"
    )

    axis.legend(
        title="FSA",
        ncol=3,
    )

    save_figure(
        figure,
        output_path,
    )

def plot_fsa_hour_heatmap(
    profile: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create an FSA × Hour heatmap of mean electricity consumption.
    """
    pivot = profile.pivot(
        index="fsa",
        columns="hour",
        values="mean",
    )

    figure, axis = plt.subplots(
        figsize=(14, 6)
    )

    image = axis.imshow(
        pivot.values,
        aspect="auto",
    )

    axis.set_xticks(
        range(len(pivot.columns))
    )
    axis.set_xticklabels(
        pivot.columns
    )

    axis.set_yticks(
        range(len(pivot.index))
    )
    axis.set_yticklabels(
        pivot.index
    )

    axis.set_xlabel("Hour")
    axis.set_ylabel("FSA")
    axis.set_title(
        "Mean electricity consumption by FSA and hour"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )
    colorbar.set_label(
        "Mean consumption (kWh)"
    )

    save_figure(
        figure,
        output_path,
    )


def run_spatial_calendar(
    frame: pd.DataFrame,
    target: str,
    reports_dir: Path,
    figures_dir: Path,
    docs_dir: Path,
) -> dict[str, pd.DataFrame]:
    spatial = spatial_summary(frame, target)
    hourly = hourly_profile_by_fsa(frame, target)
    calendar = calendar_effects(frame, target)
    seasonal_by_fsa = seasonal_profile_by_fsa(frame, target)

    save_table(spatial, reports_dir / "05_03_spatial_summary.csv")
    save_table(hourly, reports_dir / "05_03_hourly_profile_by_fsa.csv")
    save_table(seasonal_by_fsa, reports_dir / "05_03_seasonal_profile_by_fsa.csv")

    for name, table in calendar.items():
        save_table(table, reports_dir / f"05_03_calendar_{name}.csv")

    plot_spatial_summary(
        spatial,
        figures_dir / "05_03_consumption_by_fsa.png",
    )
    plot_hourly_profiles(
        hourly,
        figures_dir / "05_03_hourly_profiles_by_fsa.png",
    )
    plot_calendar_binary(
        calendar["weekend"],
        "is_weekend",
        figures_dir / "05_03_weekend_effect.png",
        "Mean consumption: weekday versus weekend",
    )
    plot_calendar_binary(
        calendar["holiday"],
        "is_public_holiday",
        figures_dir / "05_03_holiday_effect.png",
        "Mean consumption: holiday versus non-holiday",
    )
    plot_calendar_binary(
        calendar["dst_transition"],
        "is_dst_transition_day",
        figures_dir / "05_03_dst_transition_effect.png",
        "Mean consumption on DST transition dates",
    )

    plot_seasonal_profiles_by_fsa(
        seasonal_by_fsa,
        figures_dir
        / "05_03_seasonal_profiles_by_fsa.png",
    )

    plot_fsa_hour_heatmap(
        hourly,
        figures_dir
        / "05_03_fsa_hour_heatmap.png",
    )

    write_section_markdown(
        docs_dir / "05_03_Spatial_and_Calendar_Analysis.md",
        "EDA 05.03 — Spatial and Calendar Analysis",
        [
            "This section compares the six FSAs and evaluates demand differences "
            "associated with hourly profiles, weekends, holidays, seasons, quarters, "
            "workdays, long weekends, DST periods, and DST transition dates.",
        ],
        [
            ("Spatial summary", spatial),
            ("Weekend effect", calendar["weekend"]),
            ("Holiday effect", calendar["holiday"]),
            ("DST transition effect", calendar["dst_transition"]),
        ],
        [
            ("Consumption by FSA", "../../reports/figures/eda/05_03_consumption_by_fsa.png"),
            ("Hourly profiles by FSA", "../../reports/figures/eda/05_03_hourly_profiles_by_fsa.png"),
            ("Weekend effect", "../../reports/figures/eda/05_03_weekend_effect.png"),
            ("Holiday effect", "../../reports/figures/eda/05_03_holiday_effect.png"),
            ("DST transition effect", "../../reports/figures/eda/05_03_dst_transition_effect.png"),
            ("Seasonal profiles by FSA", "../../reports/figures/eda/05_03_seasonal_profiles_by_fsa.png"),
            ("FSA × Hour heatmap", "../../reports/figures/eda/05_03_fsa_hour_heatmap.png"),
        ],
    )

    return {
        "spatial": spatial,
        "hourly_by_fsa": hourly,
        "seasonal_by_fsa": seasonal_by_fsa,
        **{f"calendar_{name}": table for name, table in calendar.items()},
    }
