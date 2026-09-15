"""Outlier, extreme-weather, exploratory Peak, and feature-review analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import save_figure, save_table, write_section_markdown


def extreme_hours(
    frame: pd.DataFrame,
    target: str,
    top_n: int,
) -> pd.DataFrame:
    """Return the largest hourly demand observations."""
    columns = [
        column
        for column in [
            "fsa",
            "timestamp",
            target,
            "Temp (°C)",
            "Rel Hum (%)",
            "season",
            "hour",
            "weekday_name",
            "is_weekend",
            "is_public_holiday",
            "assigned_station_name",
        ]
        if column in frame.columns
    ]

    return (
        frame.nlargest(top_n, target)[columns]
        .sort_values(target, ascending=False)
        .reset_index(drop=True)
    )


def extreme_days(
    frame: pd.DataFrame,
    target: str,
    top_n: int,
) -> pd.DataFrame:
    """Rank FSA-days by total and maximum hourly demand."""
    daily = (
        frame.assign(date_analysis=frame["timestamp"].dt.floor("D"))
        .groupby(["fsa", "date_analysis"], observed=True)
        .agg(
            daily_total_kwh=(target, "sum"),
            daily_mean_kwh=(target, "mean"),
            maximum_hourly_kwh=(target, "max"),
            mean_temperature_c=("Temp (°C)", "mean")
            if "Temp (°C)" in frame.columns
            else (target, "size"),
        )
        .reset_index()
    )

    return daily.nlargest(top_n, "daily_total_kwh").reset_index(drop=True)


def exploratory_peak_analysis(
    frame: pd.DataFrame,
    target: str,
    percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create temporary descriptive Peak labels by FSA and season.

    This is for EDA only. The final target must be recalculated within each
    training window during modeling.
    """
    thresholds = (
        frame.groupby(["fsa", "season"], observed=True)[target]
        .quantile(percentile)
        .rename("exploratory_peak_threshold_kwh")
        .reset_index()
    )

    working = frame.merge(
        thresholds,
        on=["fsa", "season"],
        how="left",
        validate="many_to_one",
    )

    working["exploratory_peak_flag"] = (
        working[target] > working["exploratory_peak_threshold_kwh"]
    ).astype("int8")

    by_fsa = (
        working.groupby("fsa", observed=True)
        .agg(
            rows=("timestamp", "size"),
            exploratory_peak_hours=("exploratory_peak_flag", "sum"),
            exploratory_peak_pct=("exploratory_peak_flag", "mean"),
            peak_mean_consumption_kwh=(
                target,
                lambda series: series[
                    working.loc[series.index, "exploratory_peak_flag"].eq(1)
                ].mean(),
            ),
        )
        .reset_index()
    )
    by_fsa["exploratory_peak_pct"] *= 100

    peak_context = (
        working.loc[working["exploratory_peak_flag"] == 1]
        .groupby(
            [
                "fsa",
                "season",
                "hour",
                "assigned_station_name",
            ],
            observed=True,
        )
        .agg(
            peak_hours=("exploratory_peak_flag", "sum"),
            mean_peak_consumption_kwh=(target, "mean"),
            mean_temperature_c=("Temp (°C)", "mean")
            if "Temp (°C)" in working.columns
            else (target, "size"),
            mean_relative_humidity_pct=("Rel Hum (%)", "mean")
            if "Rel Hum (%)" in working.columns
            else (target, "size"),
        )
        .reset_index()
    )

    return thresholds, by_fsa, peak_context

def exploratory_peak_records(
    frame: pd.DataFrame,
    target: str,
    percentile: float,
) -> pd.DataFrame:
    """
    Return row-level exploratory Peak observations.

    Peak thresholds are calculated independently by FSA and season
    using the configured percentile.

    This dataset is for exploratory analysis only and must not be
    used directly as the final modeling target.
    """
    thresholds = (
        frame.groupby(
            ["fsa", "season"],
            observed=True,
        )[target]
        .quantile(percentile)
        .rename(
            "exploratory_peak_threshold_kwh"
        )
        .reset_index()
    )

    working = frame.merge(
        thresholds,
        on=["fsa", "season"],
        how="left",
        validate="many_to_one",
    )

    working["exploratory_peak_flag"] = (
        working[target]
        > working[
            "exploratory_peak_threshold_kwh"
        ]
    ).astype("int8")

    peak_records = working.loc[
        working["exploratory_peak_flag"] == 1
    ].copy()

    return peak_records

def peak_context_summary(
    peak_records: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Summarize the main characteristics of exploratory Peak events.
    """
    if peak_records.empty:
        return pd.DataFrame()

    most_common_hour = (
        peak_records["hour"]
        .value_counts()
        .idxmax()
    )

    most_common_month = (
        peak_records["month_name"]
        .value_counts()
        .idxmax()
    )

    most_common_weekday = (
        peak_records["weekday_name"]
        .value_counts()
        .idxmax()
    )

    most_common_season = (
        peak_records["season"]
        .value_counts()
        .idxmax()
    )

    most_common_fsa = (
        peak_records["fsa"]
        .value_counts()
        .idxmax()
    )

    summary = pd.DataFrame(
        [
            {
                "peak_records":
                    len(peak_records),

                "mean_peak_consumption_kwh":
                    peak_records[target].mean(),

                "median_peak_consumption_kwh":
                    peak_records[target].median(),

                "maximum_peak_consumption_kwh":
                    peak_records[target].max(),

                "mean_peak_temperature_c":
                    peak_records["Temp (°C)"].mean(),

                "mean_peak_relative_humidity_pct":
                    peak_records["Rel Hum (%)"].mean(),

                "most_common_peak_hour":
                    most_common_hour,

                "most_common_peak_month":
                    most_common_month,

                "most_common_peak_weekday":
                    most_common_weekday,

                "most_common_peak_season":
                    most_common_season,

                "most_common_peak_fsa":
                    str(most_common_fsa),
            }
        ]
    )

    return summary

def peak_rate_by_dimension(
    frame: pd.DataFrame,
    peak_records: pd.DataFrame,
    dimension_columns: list[str],
) -> pd.DataFrame:
    """
    Calculate exploratory Peak Rate for a given set of dimensions.

    Peak Rate is defined as:
        Peak observations / Total observations * 100

    This normalizes Peak counts by the number of available observations
    in each group and allows fair comparisons across hours, weekdays,
    months, and years.
    """

    total_counts = (
        frame.groupby(
            dimension_columns,
            observed=True,
        )
        .size()
        .rename("total_observations")
        .reset_index()
    )

    peak_counts = (
        peak_records.groupby(
            dimension_columns,
            observed=True,
        )
        .size()
        .rename("peak_observations")
        .reset_index()
    )

    result = total_counts.merge(
        peak_counts,
        on=dimension_columns,
        how="left",
        validate="one_to_one",
    )

    result["peak_observations"] = (
        result["peak_observations"]
        .fillna(0)
        .astype(int)
    )

    result["peak_rate_pct"] = (
        result["peak_observations"]
        / result["total_observations"]
        * 100
    )

    return result

def build_peak_rate_tables(
    frame: pd.DataFrame,
    peak_records: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build exploratory Peak Rate tables by hour, weekday, month, and year.
    """

    peak_rate_hour = (
        peak_rate_by_dimension(
            frame,
            peak_records,
            ["hour"],
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )

    peak_rate_weekday = (
        peak_rate_by_dimension(
            frame,
            peak_records,
            ["weekday", "weekday_name"],
        )
        .sort_values("weekday")
        .reset_index(drop=True)
    )

    peak_rate_month = (
        peak_rate_by_dimension(
            frame,
            peak_records,
            ["month", "month_name"],
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    peak_rate_year = (
        peak_rate_by_dimension(
            frame,
            peak_records,
            ["year"],
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    return {
        "hour": peak_rate_hour,
        "weekday": peak_rate_weekday,
        "month": peak_rate_month,
        "year": peak_rate_year,
    }

def build_peak_episodes(
    peak_records: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify consecutive exploratory Peak observations as Peak Episodes.

    A new episode starts when:
    - the FSA changes, or
    - the time difference between consecutive Peak observations
      is greater than one hour.
    """

    if peak_records.empty:
        return pd.DataFrame()

    working = (
        peak_records
        .sort_values(
            ["fsa", "timestamp"]
        )
        .copy()
    )

    working["previous_timestamp"] = (
        working.groupby(
            "fsa",
            observed=True,
        )["timestamp"]
        .shift(1)
    )

    working["time_gap_hours"] = (
        (
            working["timestamp"]
            - working["previous_timestamp"]
        )
        .dt.total_seconds()
        / 3600
    )

    working["new_episode"] = (
        working["previous_timestamp"].isna()
        | (working["time_gap_hours"] != 1)
    ).astype("int8")

    working["episode_id"] = (
        working.groupby(
            "fsa",
            observed=True,
        )["new_episode"]
        .cumsum()
    )

    episodes = (
        working.groupby(
            ["fsa", "episode_id"],
            observed=True,
        )
        .agg(
            episode_start=("timestamp", "min"),
            episode_end=("timestamp", "max"),
            duration_hours=("timestamp", "size"),
            maximum_consumption_kwh=(
                "total_consumption_kwh",
                "max",
            ),
            mean_consumption_kwh=(
                "total_consumption_kwh",
                "mean",
            ),
            mean_temperature_c=(
                "Temp (°C)",
                "mean",
            ),
            season=(
                "season",
                lambda series: series.mode().iloc[0],
            ),
        )
        .reset_index()
    )

    return episodes

def peak_episode_duration_summary(
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the distribution of Peak Episode durations.
    """

    if episodes.empty:
        return pd.DataFrame()

    summary = (
        episodes.groupby(
            "duration_hours",
            observed=True,
        )
        .size()
        .rename("episode_count")
        .reset_index()
        .sort_values("duration_hours")
    )

    summary["episode_pct"] = (
        summary["episode_count"]
        / summary["episode_count"].sum()
        * 100
    )

    return summary


def peak_episode_summary(
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an overall summary of exploratory Peak Episodes.
    """

    if episodes.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "episode_count":
                    len(episodes),

                "mean_episode_duration_hours":
                    episodes["duration_hours"].mean(),

                "median_episode_duration_hours":
                    episodes["duration_hours"].median(),

                "maximum_episode_duration_hours":
                    episodes["duration_hours"].max(),

                "single_hour_episodes":
                    int(
                        (
                            episodes[
                                "duration_hours"
                            ] == 1
                        ).sum()
                    ),

                "multi_hour_episodes":
                    int(
                        (
                            episodes[
                                "duration_hours"
                            ] > 1
                        ).sum()
                    ),
            }
        ]
    )


def plot_peak_thresholds_by_fsa_season(
    thresholds: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot exploratory Peak thresholds by FSA and season.
    """
    season_order = [
        "Winter",
        "Spring",
        "Summer",
        "Fall",
    ]

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    for season in season_order:
        subset = thresholds.loc[
            thresholds["season"] == season
        ].sort_values("fsa")

        axis.plot(
            subset["fsa"].astype(str),
            subset[
                "exploratory_peak_threshold_kwh"
            ],
            marker="o",
            label=season,
        )

    axis.set_xlabel("FSA")

    axis.set_ylabel(
        "Exploratory Peak threshold (kWh)"
    )

    axis.set_title(
        "Exploratory Peak thresholds by FSA and season"
    )

    axis.legend(
        title="Season"
    )

    save_figure(
        figure,
        output_path,
    )

def plot_peak_hours_by_season(
    peak_records: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot the hourly distribution of exploratory Peak events
    separately for each season.
    """
    hourly_season = (
        peak_records.groupby(
            ["hour", "season"],
            observed=True,
        )
        .size()
        .rename("peak_hours")
        .reset_index()
    )

    season_order = [
        "Winter",
        "Spring",
        "Summer",
        "Fall",
    ]

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    for season in season_order:
        subset = hourly_season.loc[
            hourly_season["season"] == season
        ].sort_values("hour")

        axis.plot(
            subset["hour"],
            subset["peak_hours"],
            marker="o",
            label=season,
        )

    axis.set_xlabel("Hour")

    axis.set_ylabel(
        "Exploratory Peak observations"
    )

    axis.set_title(
        "Exploratory Peak occurrence by hour and season"
    )

    axis.legend(
        title="Season"
    )

    save_figure(
        figure,
        output_path,
    )

def plot_temperature_extremes_by_fsa(
    temperature_extremes: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compare mean electricity consumption under cold,
    non-extreme, and hot temperature conditions by FSA.
    """
    category_order = [
        "coldest_1_percent",
        "non_extreme",
        "hottest_1_percent",
    ]

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    for category in category_order:
        subset = temperature_extremes.loc[
            temperature_extremes[
                "temperature_extreme"
            ] == category
        ].sort_values("fsa")

        axis.plot(
            subset["fsa"].astype(str),
            subset["mean_consumption_kwh"],
            marker="o",
            label=category.replace("_", " "),
        )

    axis.set_xlabel("FSA")

    axis.set_ylabel(
        "Mean electricity consumption (kWh)"
    )

    axis.set_title(
        "Electricity consumption under temperature extremes"
    )

    axis.legend(
        title="Temperature condition"
    )

    save_figure(
        figure,
        output_path,
    )


def plot_peak_hours_by_weekday(
    peak_records: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot exploratory Peak occurrence by weekday.
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

    weekday_counts = (
        peak_records["weekday_name"]
        .value_counts()
        .reindex(
            weekday_order,
            fill_value=0,
        )
        .reset_index()
    )

    weekday_counts.columns = [
        "weekday_name",
        "peak_hours",
    ]

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.bar(
        weekday_counts["weekday_name"],
        weekday_counts["peak_hours"],
    )

    axis.set_xlabel("Weekday")

    axis.set_ylabel(
        "Exploratory Peak observations"
    )

    axis.set_title(
        "Exploratory Peak occurrence by weekday"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    save_figure(
        figure,
        output_path,
    )

def plot_peak_hours_by_month(
    peak_records: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot exploratory Peak occurrence by calendar month.
    """
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    month_counts = (
        peak_records["month_name"]
        .value_counts()
        .reindex(
            month_order,
            fill_value=0,
        )
        .reset_index()
    )

    month_counts.columns = [
        "month_name",
        "peak_hours",
    ]

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.bar(
        month_counts["month_name"],
        month_counts["peak_hours"],
    )

    axis.set_xlabel("Month")

    axis.set_ylabel(
        "Exploratory Peak observations"
    )

    axis.set_title(
        "Exploratory Peak occurrence by month"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    save_figure(
        figure,
        output_path,
    )

def plot_peak_rate_by_year(
    peak_rate_year: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot the annual evolution of exploratory Peak Rate.
    """

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.plot(
        peak_rate_year["year"],
        peak_rate_year["peak_rate_pct"],
        marker="o",
    )

    axis.set_xlabel("Year")

    axis.set_ylabel(
        "Exploratory Peak Rate (%)"
    )

    axis.set_title(
        "Annual evolution of exploratory Peak Rate"
    )

    save_figure(
        figure,
        output_path,
    )


def plot_peak_rate_by_hour(
    peak_rate_hour: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot exploratory Peak Rate by hour of day.
    """

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.plot(
        peak_rate_hour["hour"],
        peak_rate_hour["peak_rate_pct"],
        marker="o",
    )

    axis.set_xlabel("Hour")

    axis.set_ylabel(
        "Exploratory Peak Rate (%)"
    )

    axis.set_title(
        "Exploratory Peak Rate by hour"
    )

    save_figure(
        figure,
        output_path,
    )

def plot_peak_rate_by_weekday(
    peak_rate_weekday: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot exploratory Peak Rate by weekday.
    """

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.bar(
        peak_rate_weekday["weekday_name"],
        peak_rate_weekday["peak_rate_pct"],
    )

    axis.set_xlabel("Weekday")

    axis.set_ylabel(
        "Exploratory Peak Rate (%)"
    )

    axis.set_title(
        "Exploratory Peak Rate by weekday"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    save_figure(
        figure,
        output_path,
    )

def plot_peak_rate_by_month(
    peak_rate_month: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot exploratory Peak Rate by calendar month.
    """

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.bar(
        peak_rate_month["month_name"],
        peak_rate_month["peak_rate_pct"],
    )

    axis.set_xlabel("Month")

    axis.set_ylabel(
        "Exploratory Peak Rate (%)"
    )

    axis.set_title(
        "Exploratory Peak Rate by month"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    save_figure(
        figure,
        output_path,
    )

def plot_peak_episode_duration(
    duration_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot the distribution of exploratory Peak Episode durations.
    """

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.bar(
        duration_summary["duration_hours"],
        duration_summary["episode_count"],
    )

    axis.set_xlabel(
        "Peak Episode duration (hours)"
    )

    axis.set_ylabel(
        "Number of Peak Episodes"
    )

    axis.set_title(
        "Distribution of exploratory Peak Episode duration"
    )

    save_figure(
        figure,
        output_path,
    )



def hot_cold_extremes(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Compare demand during descriptive heat and cold extremes."""
    if "Temp (°C)" not in frame.columns:
        return pd.DataFrame()

    working = frame[[target, "Temp (°C)", "fsa"]].dropna().copy()
    cold_threshold = working["Temp (°C)"].quantile(0.01)
    hot_threshold = working["Temp (°C)"].quantile(0.99)

    working["temperature_extreme"] = "non_extreme"
    working.loc[
        working["Temp (°C)"] <= cold_threshold,
        "temperature_extreme",
    ] = "coldest_1_percent"
    working.loc[
        working["Temp (°C)"] >= hot_threshold,
        "temperature_extreme",
    ] = "hottest_1_percent"

    return (
        working.groupby(["fsa", "temperature_extreme"], observed=True)
        .agg(
            count=(target, "count"),
            mean_consumption_kwh=(target, "mean"),
            median_consumption_kwh=(target, "median"),
            maximum_consumption_kwh=(target, "max"),
            mean_temperature_c=("Temp (°C)", "mean"),
        )
        .reset_index()
    )


def preliminary_feature_review(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Classify columns for later modeling review without deleting them."""
    rows = []

    identifiers = {
        "timestamp",
        "timestamp_utc",
        "timestamp_toronto",
        "date",
        "assigned_climate_id",
        "assigned_station_name",
    }
    target_columns = {target}
    descriptive_text = {
        "month_name",
        "weekday_name",
        "holiday_name",
        "Weather",
    }

    for column in frame.columns:
        missing_pct = frame[column].isna().mean() * 100
        unique_count = frame[column].nunique(dropna=True)

        if column in target_columns:
            recommendation = "target_keep"
            reason = "Primary forecasting target."
        elif column in identifiers:
            recommendation = "retain_for_traceability_not_directly_as_numeric_feature"
            reason = "Identifier or temporal audit field."
        elif column in descriptive_text:
            recommendation = "review_encoding_or_redundancy"
            reason = "Categorical/descriptive field may duplicate encoded calendar information."
        elif missing_pct >= 80:
            recommendation = "review_for_exclusion_or_special_handling"
            reason = "Very high missingness; usefulness must be demonstrated."
        elif missing_pct >= 50:
            recommendation = "retain_for_eda_then_review_model_specific_handling"
            reason = "Substantial missingness but potentially meaningful weather information."
        elif unique_count <= 1:
            recommendation = "exclude_constant"
            reason = "No variance."
        else:
            recommendation = "retain_candidate"
            reason = "Potentially useful predictor; final selection requires validation."

        rows.append(
            {
                "column": column,
                "dtype": str(frame[column].dtype),
                "missing_pct": round(missing_pct, 6),
                "unique_count": int(unique_count),
                "preliminary_recommendation": recommendation,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)


def plot_peak_hours(
    context: pd.DataFrame,
    output_path: Path,
) -> None:
    hourly = (
        context.groupby("hour", observed=True)["peak_hours"]
        .sum()
        .reset_index()
    )

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(hourly["hour"], hourly["peak_hours"])
    axis.set_xlabel("Hour")
    axis.set_ylabel("Exploratory Peak hours")
    axis.set_title("Exploratory Peak occurrence by hour")
    save_figure(figure, output_path)


def plot_extreme_days(
    days: pd.DataFrame,
    output_path: Path,
) -> None:
    subset = days.head(20).copy()
    labels = (
        subset["fsa"].astype(str)
        + " | "
        + subset["date_analysis"].dt.strftime("%Y-%m-%d")
    )

    figure, axis = plt.subplots(figsize=(11, 7))
    axis.barh(labels, subset["daily_total_kwh"])
    axis.invert_yaxis()
    axis.set_xlabel("Daily total consumption (kWh)")
    axis.set_ylabel("FSA and date")
    axis.set_title("Highest-demand FSA-days")
    save_figure(figure, output_path)


def run_extremes_peaks(
    frame: pd.DataFrame,
    target: str,
    percentile: float,
    top_n_hours: int,
    top_n_days: int,
    reports_dir: Path,
    figures_dir: Path,
    docs_dir: Path,
) -> dict[str, pd.DataFrame]:
    hours = extreme_hours(frame, target, top_n_hours)
    days = extreme_days(frame, target, top_n_days)
    thresholds, peaks_by_fsa, peak_context = exploratory_peak_analysis(
        frame,
        target,
        percentile,
    )

    peak_records = exploratory_peak_records(
        frame,
        target,
        percentile,
    )

    peak_summary = peak_context_summary(
        peak_records,
        target,
    )
    temperature_extremes = hot_cold_extremes(frame, target)
    feature_review = preliminary_feature_review(frame, target)

    peak_by_month = (
        peak_records.groupby(
            ["month", "month_name"],
            observed=True,
        )
        .size()
        .rename("peak_hours")
        .reset_index()
        .sort_values("month")
    )

    peak_by_weekday = (
        peak_records.groupby(
            ["weekday", "weekday_name"],
            observed=True,
        )
        .size()
        .rename("peak_hours")
        .reset_index()
        .sort_values("weekday")
    )

    peak_by_season = (
        peak_records.groupby(
            "season",
            observed=True,
        )
        .size()
        .rename("peak_hours")
        .reset_index()
    )

    peak_rates = build_peak_rate_tables(
        frame,
        peak_records,
    )

    peak_episodes = build_peak_episodes(
        peak_records,
    )

    peak_episode_duration = (
        peak_episode_duration_summary(
            peak_episodes
        )
    )

    peak_episode_overview = (
        peak_episode_summary(
            peak_episodes
        )
    )

    save_table(hours, reports_dir / "05_05_extreme_hours.csv")
    save_table(days, reports_dir / "05_05_extreme_days.csv")
    save_table(thresholds, reports_dir / "05_05_exploratory_peak_thresholds.csv")
    save_table(peaks_by_fsa, reports_dir / "05_05_exploratory_peaks_by_fsa.csv")
    save_table(peak_context, reports_dir / "05_05_exploratory_peak_context.csv")
    save_table(temperature_extremes, reports_dir / "05_05_temperature_extremes.csv")
    save_table(feature_review, reports_dir / "05_05_preliminary_feature_review.csv")
    save_table(peak_summary, reports_dir / "05_05_peak_context_summary.csv")
    save_table(peak_by_month, reports_dir / "05_05_peaks_by_month.csv")
    save_table(peak_by_weekday, reports_dir / "05_05_peaks_by_weekday.csv")
    save_table(peak_by_season, reports_dir / "05_05_peaks_by_season.csv")
    save_table(peak_rates["hour"], reports_dir / "05_05_peak_rate_by_hour.csv")
    save_table(peak_rates["weekday"], reports_dir / "05_05_peak_rate_by_weekday.csv")
    save_table(peak_rates["month"], reports_dir / "05_05_peak_rate_by_month.csv")
    save_table(peak_rates["year"], reports_dir / "05_05_peak_rate_by_year.csv")
    save_table(peak_episodes, reports_dir / "05_05_peak_episodes.csv")
    save_table(peak_episode_duration, reports_dir / "05_05_peak_episode_duration.csv")
    save_table(peak_episode_overview, reports_dir / "05_05_peak_episode_summary.csv")

    plot_peak_hours(
        peak_context,
        figures_dir / "05_05_exploratory_peaks_by_hour.png",
    )
    plot_extreme_days(
        days,
        figures_dir / "05_05_extreme_days.png",
    )

    plot_peak_thresholds_by_fsa_season(
        thresholds,
        figures_dir
        / "05_05_peak_thresholds_by_fsa_season.png",
    )

    plot_peak_hours_by_season(
        peak_records,
        figures_dir
        / "05_05_peaks_by_hour_and_season.png",
    )

    plot_temperature_extremes_by_fsa(
        temperature_extremes,
        figures_dir
        / "05_05_temperature_extremes_by_fsa.png",
    )

    plot_peak_hours_by_weekday(
        peak_records,
        figures_dir
        / "05_05_peaks_by_weekday.png",
    )

    plot_peak_hours_by_month(
        peak_records,
        figures_dir
        / "05_05_peaks_by_month.png",
    )

    plot_peak_rate_by_hour(
        peak_rates["hour"],
        figures_dir
        / "05_05_peak_rate_by_hour.png",
    )

    plot_peak_rate_by_weekday(
        peak_rates["weekday"],
        figures_dir
        / "05_05_peak_rate_by_weekday.png",
    )

    plot_peak_rate_by_month(
        peak_rates["month"],
        figures_dir
        / "05_05_peak_rate_by_month.png",
    )

    plot_peak_rate_by_year(
        peak_rates["year"],
        figures_dir
        / "05_05_peak_rate_by_year.png",
    )

    plot_peak_episode_duration(
        peak_episode_duration,
        figures_dir
        / "05_05_peak_episode_duration.png",
    )

    write_section_markdown(
        docs_dir / "05_05_Extremes_Peaks_and_Feature_Review.md",
        "EDA 05.05 — Extremes, Exploratory Peaks, and Feature Review",
        [
            f"Exploratory Peak hours are defined temporarily using the "
            f"{percentile:.3f} quantile by FSA and season. This definition is "
            "descriptive only and must not be reused as the final modeling target "
            "without recalculating thresholds inside each training window.",
        ],
        [
            ("Highest-demand hours", hours),
            ("Highest-demand FSA-days", days),
            ("Exploratory Peak thresholds", thresholds),
            ("Exploratory Peak counts by FSA", peaks_by_fsa),
            ("Temperature extremes", temperature_extremes),
            ("Preliminary feature review", feature_review),
        ],
        [
            ("Exploratory Peaks by hour", "../../reports/figures/eda/05_05_exploratory_peaks_by_hour.png"),
            ("Highest-demand FSA-days", "../../reports/figures/eda/05_05_extreme_days.png"),
            ("Exploratory Peak thresholds by FSA and season", "../../reports/figures/eda/05_05_peak_thresholds_by_fsa_season.png"),
            ("Exploratory Peaks by hour and season", "../../reports/figures/eda/05_05_peaks_by_hour_and_season.png"),
            ("Temperature extremes by FSA", "../../reports/figures/eda/05_05_temperature_extremes_by_fsa.png"),
            ("Exploratory Peaks by weekday", "../../reports/figures/eda/05_05_peaks_by_weekday.png"),
            ("Exploratory Peaks by month", "../../reports/figures/eda/05_05_peaks_by_month.png"),
            ("Exploratory Peak Rate by hour", "../../reports/figures/eda/05_05_peak_rate_by_hour.png"),
            ("Exploratory Peak Rate by weekday", "../../reports/figures/eda/05_05_peak_rate_by_weekday.png"),
            ("Exploratory Peak Rate by month", "../../reports/figures/eda/05_05_peak_rate_by_month.png"),
            ("Exploratory Peak Rate by year", "../../reports/figures/eda/05_05_peak_rate_by_year.png"),
            ("Exploratory Peak Episode duration distribution", "../../reports/figures/eda/05_05_peak_episode_duration.png")
        ],
    )

    return {
        "extreme_hours": hours,
        "extreme_days": days,
        "peak_thresholds": thresholds,
        "peaks_by_fsa": peaks_by_fsa,
        "peak_context": peak_context,
        "peak_summary": peak_summary,

        "peaks_by_month": peak_by_month,
        "peaks_by_weekday": peak_by_weekday,
        "peaks_by_season": peak_by_season,

        "peak_rate_by_hour":
            peak_rates["hour"],

        "peak_rate_by_weekday":
            peak_rates["weekday"],

        "peak_rate_by_month":
            peak_rates["month"],

        "peak_rate_by_year":
            peak_rates["year"],

        "peak_episodes":
            peak_episodes,

        "peak_episode_duration":
            peak_episode_duration,

        "peak_episode_summary":
            peak_episode_overview,

        "temperature_extremes":
            temperature_extremes,

        "feature_review":
            feature_review,
    }
