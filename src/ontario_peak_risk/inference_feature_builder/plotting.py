"""Diagnostic figures for the IFB."""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt


def demand_recency_plot(coverage: pd.DataFrame):    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(coverage["fsa"], coverage["gap_from_latest_to_origin_hours"])
    ax.axhline(0, linestyle="--")
    ax.set_ylabel("Gap to forecast origin (hours)")
    ax.set_title("Operational Demand Recency by FSA")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def missing_demand_plot(coverage: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(coverage["fsa"], coverage["missing_hours"])
    ax.set_ylabel("Missing hourly demand observations")
    ax.set_title("Required Recent Demand Coverage")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def weather_coverage_plot(coverage: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(coverage["fsa"], coverage["available_complete_rows"])
    ax.axhline(25, linestyle="--", label="Required h0...h24")
    ax.set_ylabel("Complete weather rows")
    ax.set_title("Weather Forecast Coverage by FSA")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def lag_profile_plot(generated: pd.DataFrame, fsa: str):
    subset = generated.loc[generated["fsa"].eq(fsa)].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in ["target_lag_24h", "target_lag_48h", "target_lag_168h"]:
        ax.plot(subset["horizon"], subset[col], marker="o", label=col)
    ax.set_xticks(range(1, 25))
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Historical consumption (kWh)")
    ax.set_title(f"Target-Relative Demand Lags — {fsa}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def feature_completeness_heatmap(generated: pd.DataFrame, features: list[str]):
    matrix = (
        generated.assign(
            row_key=generated["fsa"]
            + "_h"
            + generated["horizon"].astype(str).str.zfill(2)
        )
        .set_index("row_key")[features]
        .notna()
        .astype(int)
    )
    fig, ax = plt.subplots(figsize=(max(10, len(features) * 0.5), 10))
    image = ax.imshow(matrix.values, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=6)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_title("IFB Feature Completeness (1 = available)")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    return fig


def replay_match_rate_plot(summary: pd.DataFrame):
    grouped = (
        summary.groupby(["model", "horizon"], observed=True)["match_rate_pct"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for model, group in grouped.groupby("model", observed=True):
        ax.plot(
            group["horizon"], group["match_rate_pct"],
            marker="o", label=model.upper()
        )
    ax.set_xticks(range(1, 25))
    ax.set_ylim(0, 101)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Mean feature match rate (%)")
    ax.set_title("Historical Replay — IFB vs Original Feature Builder")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def domain_warning_plot(domain_report: pd.DataFrame):
    if domain_report.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No out-of-domain warnings", ha="center", va="center")
        ax.axis("off")
        return fig

    counts = (
        domain_report["domain_status"].value_counts()
        .rename_axis("status").reset_index(name="count")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts["status"], counts["count"])
    ax.set_ylabel("Input observations")
    ax.set_title("Operational Input Domain Warnings")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def forecast_profile_plot(prediction: pd.DataFrame):
    """24-hour electricity-demand forecast by FSA."""
    date_label = _prediction_date_label(prediction)
    fig, ax = plt.subplots(figsize=(12, 6))
    for fsa, group in prediction.groupby("fsa", observed=True):
        ordered = group.sort_values("horizon")
        ax.plot(
            ordered["horizon"],
            ordered["forecast_consumption_kwh"],
            marker="o",
            label=str(fsa),
        )
    ax.set_xticks(range(1, 25))
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Forecast consumption (kWh)")
    # ax.set_title("24-Hour Electricity-Demand Forecast by FSA")
    ax.set_title(f"24-Hour Electricity-Demand Forecast by FSA — {date_label}")
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def peak_risk_profile_plot(
    prediction: pd.DataFrame,
    threshold: float,
):
    """Peak-Risk score by forecast horizon and FSA."""
    date_label = _prediction_date_label(prediction)
    fig, ax = plt.subplots(figsize=(12, 6))
    for fsa, group in prediction.groupby("fsa", observed=True):
        ordered = group.sort_values("horizon")
        ax.plot(
            ordered["horizon"],
            ordered["peak_risk_score"],
            marker="o",
            label=str(fsa),
        )
    ax.axhline(
        float(threshold),
        linestyle="--",
        label=f"Alert threshold = {float(threshold):.2f}",
    )
    ax.set_xticks(range(1, 25))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Peak-Risk score")
    # ax.set_title("24-Hour Peak-Risk Profile by FSA")
    ax.set_title(f"24-Hour Peak-Risk Profile by FSA — {date_label}")
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def peak_risk_heatmap(prediction: pd.DataFrame):
    """FSA x horizon heatmap of Peak-Risk scores."""
    date_label = _prediction_date_label(prediction)
    matrix = (
        prediction.pivot(
            index="fsa",
            columns="horizon",
            values="peak_risk_score",
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(13, 4.5))
    image = ax.imshow(matrix.values, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("FSA")
    # ax.set_title("Peak-Risk Heatmap")
    ax.set_title(f"Peak-Risk Heatmap — {date_label}")
    fig.colorbar(image, ax=ax, label="Peak-Risk score")
    fig.tight_layout()
    return fig


def actual_vs_forecast_plot(evaluation: pd.DataFrame):
    """Observed versus forecast consumption when post-origin actuals exist."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for fsa, group in evaluation.groupby("fsa", observed=True):
        ordered = group.sort_values("horizon")
        ax.plot(
            ordered["horizon"],
            ordered["actual_consumption_kwh"],
            marker="o",
            label=f"{fsa} actual",
        )
        ax.plot(
            ordered["horizon"],
            ordered["forecast_consumption_kwh"],
            linestyle="--",
            label=f"{fsa} forecast",
        )
    ax.set_xticks(range(1, 25))
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Consumption (kWh)")
    ax.set_title("Post-Holdout Operational Simulation: Actual vs Forecast")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig

def _prediction_date_label(prediction: pd.DataFrame) -> str:
    dates = pd.to_datetime(
        prediction["target_timestamp"]
    ).dt.normalize().drop_duplicates()

    if len(dates) == 1:
        return dates.iloc[0].strftime("%B %d, %Y")

    return (
        f"{dates.min().strftime('%B %d, %Y')} – "
        f"{dates.max().strftime('%B %d, %Y')}"
    )


def forecast_with_peak_alerts_plot(prediction: pd.DataFrame):
    """Demand forecast with Peak-Risk alert markers."""

    date_label = _prediction_date_label(prediction)

    fig, ax = plt.subplots(figsize=(13, 6))

    for fsa, group in prediction.groupby("fsa", observed=True):
        group = group.sort_values("target_timestamp")

        ax.plot(
            group["target_timestamp"],
            group["forecast_consumption_kwh"],
            marker="o",
            label=fsa,
        )

        alerts = group.loc[group["peak_alert"]]

        ax.scatter(
            alerts["target_timestamp"],
            alerts["forecast_consumption_kwh"],
            marker="s",
            s=75,
        )

    ax.set_xlabel("Forecast hour")
    ax.set_ylabel("Forecast consumption (kWh)")
    ax.set_title(
        f"24-Hour Demand Forecast and Peak-Risk Alerts — {date_label}"
    )

    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig
