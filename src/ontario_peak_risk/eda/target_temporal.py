"""Target distribution, trend, seasonality, profiles, and autocorrelation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import save_figure, save_table, write_section_markdown


def target_statistics(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Overall descriptive statistics for electricity consumption."""
    series = frame[target].dropna()

    return pd.DataFrame(
        [
            {
                "count": len(series),
                "mean": series.mean(),
                "median": series.median(),
                "standard_deviation": series.std(),
                "minimum": series.min(),
                "p01": series.quantile(0.01),
                "p05": series.quantile(0.05),
                "p25": series.quantile(0.25),
                "p75": series.quantile(0.75),
                "p95": series.quantile(0.95),
                "p975": series.quantile(0.975),
                "p99": series.quantile(0.99),
                "maximum": series.max(),
                "skewness": series.skew(),
            }
        ]
    )


def target_by_fsa(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Target distribution by FSA."""
    return (
        frame.groupby("fsa", observed=True)[target]
        .agg(
            count="count",
            mean="mean",
            median="median",
            standard_deviation="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )


def temporal_profiles(
    frame: pd.DataFrame,
    target: str,
) -> dict[str, pd.DataFrame]:
    """
    Build yearly, monthly, weekday, hourly, seasonal, and daily
    electricity-consumption profiles.

    Temporal categories are explicitly sorted to preserve their
    chronological order in tables and visualizations.
    """

    # ------------------------------------------------------------------
    # Yearly profile
    # ------------------------------------------------------------------
    yearly_profile = (
        frame.groupby(
            "year",
            observed=True,
        )[target]
        .agg(["mean", "median", "max"])
        .reset_index()
        .sort_values("year")
    )

    # ------------------------------------------------------------------
    # Monthly profile
    # ------------------------------------------------------------------
    monthly_profile = (
        frame.groupby(
            ["month", "month_name"],
            observed=True,
        )[target]
        .agg(["mean", "median", "max"])
        .reset_index()
        .sort_values("month")
    )

    # ------------------------------------------------------------------
    # Weekday profile
    # Monday = 0 ... Sunday = 6
    # ------------------------------------------------------------------
    weekday_profile = (
        frame.groupby(
            ["weekday", "weekday_name"],
            observed=True,
        )[target]
        .agg(["mean", "median", "max"])
        .reset_index()
        .sort_values("weekday")
    )

    # ------------------------------------------------------------------
    # Hourly profile
    # ------------------------------------------------------------------
    hourly_profile = (
        frame.groupby(
            "hour",
            observed=True,
        )[target]
        .agg(["mean", "median", "max"])
        .reset_index()
        .sort_values("hour")
    )

    # ------------------------------------------------------------------
    # Seasonal profile
    # Explicit chronological meteorological order
    # ------------------------------------------------------------------
    season_order = [
        "Winter",
        "Spring",
        "Summer",
        "Fall",
    ]

    seasonal_data = frame.copy()

    seasonal_data["season"] = pd.Categorical(
        seasonal_data["season"],
        categories=season_order,
        ordered=True,
    )

    seasonal_profile = (
        seasonal_data.groupby(
            "season",
            observed=True,
        )[target]
        .agg(["mean", "median", "max"])
        .reset_index()
        .sort_values("season")
    )

    # ------------------------------------------------------------------
    # Daily profile
    # ------------------------------------------------------------------
    daily_profile = (
        frame.assign(
            date_analysis=frame["timestamp"].dt.floor("D")
        )
        .groupby(
            "date_analysis",
            observed=True,
        )[target]
        .agg(
            total="sum",
            mean="mean",
            maximum="max",
        )
        .reset_index()
        .sort_values("date_analysis")
    )

    return {
        "year": yearly_profile,
        "month": monthly_profile,
        "weekday": weekday_profile,
        "hour": hourly_profile,
        "season": seasonal_profile,
        "daily": daily_profile,
    }


def autocorrelation_summary(
    frame: pd.DataFrame,
    target: str,
    lags: list[int],
) -> pd.DataFrame:
    """Calculate target autocorrelation by FSA for selected lags."""
    rows = []

    for fsa, group in frame.groupby("fsa", observed=True, sort=True):
        series = (
            group.sort_values("timestamp")
            .set_index("timestamp")[target]
        )

        for lag in lags:
            rows.append(
                {
                    "fsa": str(fsa),
                    "lag_hours": lag,
                    "autocorrelation": series.autocorr(lag=lag),
                }
            )

    return pd.DataFrame(rows)


def plot_target_distribution(
    frame: pd.DataFrame,
    target: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(frame[target].dropna(), bins=60)
    axis.set_xlabel("Electricity consumption (kWh)")
    axis.set_ylabel("Hourly observations")
    axis.set_title("Distribution of hourly electricity consumption")
    save_figure(figure, output_path)


def plot_daily_trend(
    daily: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(daily["date_analysis"], daily["mean"])
    axis.set_xlabel("Date")
    axis.set_ylabel("Mean hourly consumption (kWh)")
    axis.set_title("Daily mean electricity consumption")
    save_figure(figure, output_path)


def plot_profile(
    profile: pd.DataFrame,
    x: str,
    output_path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(profile[x], profile["mean"], marker="o")
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel("Mean consumption (kWh)")
    axis.set_title(title)

    if profile[x].dtype == "object" or str(profile[x].dtype) == "category":
        axis.tick_params(axis="x", rotation=45)

    save_figure(figure, output_path)


def plot_hourly_boxplot(
    frame: pd.DataFrame,
    target: str,
    output_path: Path,
) -> None:
    """
    Create a boxplot of electricity consumption by hour.

    The chart shows the median, interquartile range, dispersion,
    and potential outliers for each hour of the day.
    """
    hourly_data = [
        frame.loc[
            frame["hour"] == hour,
            target,
        ].dropna().values
        for hour in sorted(frame["hour"].dropna().unique())
    ]

    hour_labels = sorted(
        frame["hour"].dropna().unique()
    )

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    axis.boxplot(
        hourly_data,
        labels=hour_labels,
        showfliers=False,
    )

    axis.set_xlabel("Hour")
    axis.set_ylabel(
        "Electricity consumption (kWh)"
    )
    axis.set_title(
        "Distribution of electricity consumption by hour"
    )

    save_figure(
        figure,
        output_path,
    )

def plot_weekday_boxplot(
    frame: pd.DataFrame,
    target: str,
    output_path: Path,
) -> None:
    """
    Create a boxplot of electricity consumption by weekday.

    Weekdays are displayed in chronological order from
    Monday through Sunday.
    """
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday_data = [
        frame.loc[
            frame["weekday_name"] == weekday,
            target,
        ].dropna().values
        for weekday in weekday_order
    ]

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    axis.boxplot(
        weekday_data,
        labels=weekday_order,
        showfliers=False,
    )

    axis.set_xlabel("Weekday")
    axis.set_ylabel(
        "Electricity consumption (kWh)"
    )
    axis.set_title(
        "Distribution of electricity consumption by weekday"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    save_figure(
        figure,
        output_path,
    )

def plot_autocorrelation_by_fsa(
    autocorrelation: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot selected electricity-demand autocorrelations by FSA.

    Each line represents one FSA and shows how strongly current
    electricity demand is related to previous observations.
    """
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for fsa, group in autocorrelation.groupby(
        "fsa",
        observed=True,
    ):
        group = group.sort_values(
            "lag_hours"
        )

        axis.plot(
            group["lag_hours"],
            group["autocorrelation"],
            marker="o",
            label=str(fsa),
        )

    axis.set_xlabel(
        "Lag (hours)"
    )

    axis.set_ylabel(
        "Autocorrelation"
    )

    axis.set_title(
        "Electricity demand autocorrelation by FSA"
    )

    axis.set_ylim(
        -1,
        1,
    )

    axis.axhline(
        y=0,
        linewidth=1,
    )

    axis.legend(
        title="FSA"
    )

    save_figure(
        figure,
        output_path,
    )

def hour_month_heatmap_data(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Calculate mean electricity consumption for every
    month-hour combination.
    """
    heatmap_data = (
        frame.groupby(
            ["month", "hour"],
            observed=True,
        )[target]
        .mean()
        .reset_index(
            name="mean_consumption_kwh"
        )
        .sort_values(
            ["month", "hour"]
        )
    )

    return heatmap_data

def plot_hour_month_heatmap(
    heatmap_data: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create a Month × Hour heatmap of mean electricity consumption.
    """
    pivot = heatmap_data.pivot(
        index="month",
        columns="hour",
        values="mean_consumption_kwh",
    )

    figure, axis = plt.subplots(
        figsize=(14, 7)
    )

    image = axis.imshow(
        pivot.values,
        aspect="auto",
        origin="upper",
    )

    axis.set_xticks(
        range(len(pivot.columns))
    )

    axis.set_xticklabels(
        pivot.columns
    )

    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    axis.set_yticks(
        range(len(pivot.index))
    )

    axis.set_yticklabels(
        [
            month_labels[int(month) - 1]
            for month in pivot.index
        ]
    )

    axis.set_xlabel(
        "Hour"
    )

    axis.set_ylabel(
        "Month"
    )

    axis.set_title(
        "Mean electricity consumption by month and hour"
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

def run_target_temporal(
    frame: pd.DataFrame,
    target: str,
    lags: list[int],
    reports_dir: Path,
    figures_dir: Path,
    docs_dir: Path,
) -> dict[str, pd.DataFrame]:
    statistics = target_statistics(frame, target)
    by_fsa = target_by_fsa(frame, target)
    profiles = temporal_profiles(frame, target)
    autocorrelation = autocorrelation_summary(frame, target, lags)

    heatmap_data = hour_month_heatmap_data(frame, target)

    

    save_table(statistics, reports_dir / "05_02_target_statistics.csv")
    save_table(by_fsa, reports_dir / "05_02_target_by_fsa.csv")
    save_table(autocorrelation, reports_dir / "05_02_autocorrelation.csv")
    save_table(heatmap_data, reports_dir / "05_02_hour_month_heatmap.csv",
)

    for name, profile in profiles.items():
        save_table(profile, reports_dir / f"05_02_profile_{name}.csv")

    plot_target_distribution(
        frame,
        target,
        figures_dir / "05_02_target_distribution.png",
    )
    plot_daily_trend(
        profiles["daily"],
        figures_dir / "05_02_daily_trend.png",
    )
    plot_profile(
        profiles["hour"],
        "hour",
        figures_dir / "05_02_hourly_profile.png",
        "Mean consumption by hour",
    )
    plot_profile(
        profiles["month"],
        "month_name",
        figures_dir / "05_02_monthly_profile.png",
        "Mean consumption by month",
    )
    plot_profile(
        profiles["weekday"],
        "weekday_name",
        figures_dir / "05_02_weekday_profile.png",
        "Mean consumption by weekday",
    )
    plot_profile(
        profiles["season"],
        "season",
        figures_dir / "05_02_seasonal_profile.png",
        "Mean consumption by season",
    )

    plot_hourly_boxplot(
        frame,
        target,
        figures_dir
        / "05_02_hourly_boxplot.png",
    )

    plot_weekday_boxplot(
        frame,
        target,
        figures_dir
        / "05_02_weekday_boxplot.png",
    )

    plot_autocorrelation_by_fsa(
        autocorrelation,
        figures_dir
        / "05_02_autocorrelation_by_fsa.png",
    )

    plot_hour_month_heatmap(
        heatmap_data,
        figures_dir
        / "05_02_hour_month_heatmap.png",
    )

    write_section_markdown(
        docs_dir / "05_02_Target_and_Temporal_Analysis.md",
        "EDA 05.02 — Target and Temporal Analysis",
        [
            "This section examines the distribution of `total_consumption_kwh`, "
            "long-term trends, annual, monthly, weekly, daily, and hourly patterns, "
            "and selected target autocorrelations.",
        ],
        [
            ("Target statistics", statistics),
            ("Target statistics by FSA", by_fsa),
            ("Selected autocorrelations", autocorrelation),
        ],
        [
            ("Target distribution", "../../reports/figures/eda/05_02_target_distribution.png"),
            ("Daily trend", "../../reports/figures/eda/05_02_daily_trend.png"),
            ("Hourly profile", "../../reports/figures/eda/05_02_hourly_profile.png"),
            ("Monthly profile", "../../reports/figures/eda/05_02_monthly_profile.png"),
            ("Weekday profile", "../../reports/figures/eda/05_02_weekday_profile.png"),
            ("Seasonal profile", "../../reports/figures/eda/05_02_seasonal_profile.png"),
            ("Hourly consumption distribution","../../reports/figures/eda/05_02_hourly_boxplot.png"),
            ("Weekday consumption distribution","../../reports/figures/eda/05_02_weekday_boxplot.png"),
            ("Autocorrelation by FSA","../../reports/figures/eda/05_02_autocorrelation_by_fsa.png"),
            ("Month-hour demand heatmap","../../reports/figures/eda/05_02_hour_month_heatmap.png"),
        ],
    )

    return {
        "statistics": statistics,
        "by_fsa": by_fsa,
        "autocorrelation": autocorrelation,
        "hour_month_heatmap": heatmap_data,
        **profiles,
    }
