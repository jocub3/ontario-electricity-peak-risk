"""Weather coverage, missingness patterns, demand relationships, and correlations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    correlation_strength,
    save_figure,
    save_table,
    write_section_markdown,
)


def weather_coverage(
    frame: pd.DataFrame,
    weather_columns: list[str],
) -> pd.DataFrame:
    """Coverage by station and weather variable."""
    rows = []

    grouping = ["assigned_climate_id", "assigned_station_name"]

    for keys, group in frame.groupby(grouping, observed=True, dropna=False):
        climate_id, station_name = keys

        for column in weather_columns:
            if column not in group.columns:
                continue

            rows.append(
                {
                    "assigned_climate_id": climate_id,
                    "assigned_station_name": station_name,
                    "column": column,
                    "rows": len(group),
                    "non_null_count": int(group[column].notna().sum()),
                    "missing_count": int(group[column].isna().sum()),
                    "coverage_pct": round(group[column].notna().mean() * 100, 6),
                }
            )

    return pd.DataFrame(rows)


def weather_coverage_by_fsa_period(
    frame: pd.DataFrame,
    weather_columns: list[str],
) -> pd.DataFrame:
    """Core weather coverage by FSA and year."""
    rows = []

    for (fsa, year), group in frame.groupby(
        ["fsa", "year"],
        observed=True,
        sort=True,
    ):
        for column in weather_columns:
            if column not in group.columns:
                continue

            rows.append(
                {
                    "fsa": str(fsa),
                    "year": year,
                    "column": column,
                    "coverage_pct": round(group[column].notna().mean() * 100, 6),
                }
            )

    return pd.DataFrame(rows)


def missingness_overlap(
    frame: pd.DataFrame,
    weather_columns: list[str],
) -> pd.DataFrame:
    """Quantify pairwise overlap of missing weather observations."""
    existing = [column for column in weather_columns if column in frame.columns]
    rows = []

    for left in existing:
        for right in existing:
            both_missing = frame[left].isna() & frame[right].isna()
            rows.append(
                {
                    "variable_1": left,
                    "variable_2": right,
                    "both_missing_count": int(both_missing.sum()),
                    "both_missing_pct": round(both_missing.mean() * 100, 6),
                }
            )

    return pd.DataFrame(rows)


def target_weather_correlations(
    frame: pd.DataFrame,
    target: str,
    weather_columns: list[str],
) -> pd.DataFrame:
    """Pearson and Spearman target correlations using pairwise complete rows."""
    rows = []

    for column in weather_columns:
        if column not in frame.columns:
            continue

        pair = frame[[target, column]].dropna()

        if pair.empty or pair[column].nunique() <= 1:
            continue

        pearson = pair[target].corr(pair[column], method="pearson")
        spearman = pair[target].corr(pair[column], method="spearman")

        rows.append(
            {
                "weather_variable": column,
                "pairwise_rows": len(pair),
                "coverage_pct": round(len(pair) / len(frame) * 100, 6),
                "pearson_correlation": pearson,
                "pearson_strength": correlation_strength(pearson),
                "spearman_correlation": spearman,
                "spearman_strength": correlation_strength(spearman),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "spearman_correlation",
            key=lambda series: series.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )


def weather_correlation_matrix(
    frame: pd.DataFrame,
    target: str,
    weather_columns: list[str],
) -> pd.DataFrame:
    existing = [
        column
        for column in [target, *weather_columns]
        if column in frame.columns
    ]
    return frame[existing].corr(numeric_only=True)


def temperature_bands(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Descriptive demand by observed temperature bands."""
    if "Temp (°C)" not in frame.columns:
        return pd.DataFrame()

    working = frame[[target, "Temp (°C)"]].dropna().copy()
    working["temperature_band"] = pd.cut(
        working["Temp (°C)"],
        bins=[-np.inf, -10, 0, 10, 20, 30, np.inf],
        labels=[
            "Below -10",
            "-10 to 0",
            "0 to 10",
            "10 to 20",
            "20 to 30",
            "Above 30",
        ],
    )

    return (
        working.groupby("temperature_band", observed=True)[target]
        .agg(count="count", mean="mean", median="median", maximum="max")
        .reset_index()
    )


def plot_coverage(
    coverage: pd.DataFrame,
    output_path: Path,
) -> None:
    pivot = coverage.pivot(
        index="column",
        columns="assigned_station_name",
        values="coverage_pct",
    )

    figure, axis = plt.subplots(figsize=(11, max(5, len(pivot) * 0.4)))
    pivot.plot(kind="barh", ax=axis)
    axis.set_xlabel("Coverage (%)")
    axis.set_ylabel("Weather variable")
    axis.set_title("Weather-variable coverage by station")
    axis.legend(title="Station")
    save_figure(figure, output_path)


def plot_temperature_relationship(
    frame: pd.DataFrame,
    target: str,
    output_path: Path,
) -> None:
    subset = frame[[target, "Temp (°C)"]].dropna()

    # Sampling is for plotting only; all rows remain in tabular calculations.
    sample = subset.sample(
        n=min(20000, len(subset)),
        random_state=42,
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(sample["Temp (°C)"], sample[target], alpha=0.18, s=8)
    axis.set_xlabel("Temperature (°C)")
    axis.set_ylabel("Electricity consumption (kWh)")
    axis.set_title("Electricity consumption versus temperature")
    save_figure(figure, output_path)


def plot_correlation_matrix(
    correlation: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(correlation.values, aspect="auto", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_yticks(range(len(correlation.index)))
    axis.set_xticklabels(correlation.columns, rotation=90)
    axis.set_yticklabels(correlation.index)
    axis.set_title("Correlation matrix: target and weather variables")
    figure.colorbar(image, ax=axis)
    save_figure(figure, output_path)


def variable_retention_review(
    coverage: pd.DataFrame,
    correlations: pd.DataFrame,
    high_missing_threshold: float,
    minimum_coverage: float,
) -> pd.DataFrame:
    """Create a non-binding review table for later feature selection."""
    overall_coverage = (
        coverage.groupby("column", observed=True)["coverage_pct"]
        .mean()
        .rename("mean_coverage_pct")
        .reset_index()
    )

    output = overall_coverage.merge(
        correlations[
            [
                "weather_variable",
                "spearman_correlation",
                "spearman_strength",
            ]
        ],
        left_on="column",
        right_on="weather_variable",
        how="left",
    ).drop(columns="weather_variable")

    def recommendation(row: pd.Series) -> str:
        coverage_value = row["mean_coverage_pct"]
        correlation = row["spearman_correlation"]

        if coverage_value < minimum_coverage:
            return "review_for_exclusion_low_coverage"
        if coverage_value < (100 - high_missing_threshold):
            return "retain_for_eda_review_before_modeling"
        if pd.notna(correlation) and abs(correlation) >= 0.20:
            return "retain_candidate"
        return "retain_pending_multivariate_review"

    output["eda_recommendation"] = output.apply(recommendation, axis=1)
    return output

def plot_temperature_bands(
    temperature_data: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot mean electricity consumption across temperature bands.
    """
    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.plot(
        temperature_data["temperature_band"].astype(str),
        temperature_data["mean"],
        marker="o",
    )

    axis.set_xlabel(
        "Temperature band (°C)"
    )

    axis.set_ylabel(
        "Mean electricity consumption (kWh)"
    )

    axis.set_title(
        "Mean electricity consumption by temperature band"
    )

    axis.tick_params(
        axis="x",
        rotation=30,
    )

    save_figure(
        figure,
        output_path,
    )

def plot_weather_coverage_heatmap(
    coverage: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create a heatmap showing weather-variable coverage by station.
    """
    pivot = coverage.pivot(
        index="column",
        columns="assigned_station_name",
        values="coverage_pct",
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    image = axis.imshow(
        pivot.values,
        aspect="auto",
        vmin=0,
        vmax=100,
    )

    axis.set_xticks(
        range(len(pivot.columns))
    )

    axis.set_xticklabels(
        pivot.columns,
    )

    axis.set_yticks(
        range(len(pivot.index))
    )

    axis.set_yticklabels(
        pivot.index,
    )

    axis.set_xlabel(
        "Weather station"
    )

    axis.set_ylabel(
        "Weather variable"
    )

    axis.set_title(
        "Weather-variable coverage by station"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Coverage (%)"
    )

    save_figure(
        figure,
        output_path,
    )


def run_weather_missing(
    frame: pd.DataFrame,
    target: str,
    weather_columns: list[str],
    high_missing_threshold: float,
    minimum_coverage: float,
    reports_dir: Path,
    figures_dir: Path,
    docs_dir: Path,
) -> dict[str, pd.DataFrame]:
    coverage = weather_coverage(frame, weather_columns)
    coverage_period = weather_coverage_by_fsa_period(frame, weather_columns)
    overlap = missingness_overlap(frame, weather_columns)
    correlations = target_weather_correlations(frame, target, weather_columns)
    matrix = weather_correlation_matrix(frame, target, weather_columns)
    temperature = temperature_bands(frame, target)
    retention = variable_retention_review(
        coverage,
        correlations,
        high_missing_threshold,
        minimum_coverage,
    )

    save_table(coverage, reports_dir / "05_04_weather_coverage_by_station.csv")
    save_table(coverage_period, reports_dir / "05_04_weather_coverage_by_fsa_year.csv")
    save_table(overlap, reports_dir / "05_04_missingness_overlap.csv")
    save_table(correlations, reports_dir / "05_04_target_weather_correlations.csv")
    save_table(matrix.reset_index(), reports_dir / "05_04_weather_correlation_matrix.csv")
    save_table(temperature, reports_dir / "05_04_temperature_bands.csv")
    save_table(retention, reports_dir / "05_04_weather_variable_retention_review.csv")

    plot_coverage(
        coverage,
        figures_dir / "05_04_weather_coverage_by_station.png",
    )
    plot_temperature_relationship(
        frame,
        target,
        figures_dir / "05_04_consumption_vs_temperature.png",
    )
    plot_correlation_matrix(
        matrix,
        figures_dir / "05_04_weather_correlation_matrix.png",
    )
    plot_temperature_bands(
        temperature,
        figures_dir
        / "05_04_consumption_by_temperature_band.png",
    )

    plot_weather_coverage_heatmap(
        coverage,
        figures_dir
        / "05_04_weather_coverage_heatmap.png",
    )
    
    write_section_markdown(
        docs_dir / "05_04_Weather_Correlation_and_Missingness.md",
        "EDA 05.04 — Weather, Correlation, and Missingness",
        [
            "This section examines weather coverage by station, FSA, and year; "
            "missingness overlap; demand-weather relationships; and a preliminary "
            "retention review. No variable is automatically removed.",
        ],
        [
            ("Weather coverage by station", coverage),
            ("Target-weather correlations", correlations),
            ("Temperature bands", temperature),
            ("Preliminary weather-variable retention review", retention),
        ],
        [
            ("Weather coverage by station", "../../reports/figures/eda/05_04_weather_coverage_by_station.png"),
            ("Consumption versus temperature", "../../reports/figures/eda/05_04_consumption_vs_temperature.png"),
            ("Weather correlation matrix", "../../reports/figures/eda/05_04_weather_correlation_matrix.png"),
            ("Consumption by temperature band", "../../reports/figures/eda/05_04_consumption_by_temperature_band.png"),
            ("Weather coverage heatmap", "../../reports/figures/eda/05_04_weather_coverage_heatmap.png"),
        ],
    )

    return {
        "coverage": coverage,
        "coverage_period": coverage_period,
        "overlap": overlap,
        "correlations": correlations,
        "correlation_matrix": matrix,
        "temperature_bands": temperature,
        "retention_review": retention,
    }
