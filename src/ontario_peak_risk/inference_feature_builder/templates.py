"""Operational CSV template generation."""

from __future__ import annotations
from pathlib import Path
import pandas as pd


def generate_templates(
    templates_dir: Path,
    *,
    forecast_origin: str | pd.Timestamp,
    fsas: list[str],
) -> dict[str, Path]:
    templates_dir.mkdir(parents=True, exist_ok=True)
    origin = pd.Timestamp(forecast_origin)

    demand_path = templates_dir / "demand_update_template.csv"
    weather_history_path = templates_dir / "weather_history_update_template.csv"
    weather_forecast_path = templates_dir / "weather_forecast_template.csv"

    pd.DataFrame(columns=[
        "timestamp", "fsa", "total_consumption_kwh", "reported_premise_count"
    ]).to_csv(demand_path, index=False)

    pd.DataFrame(columns=[
        "timestamp", "fsa", "Temp (°C)", "Rel Hum (%)", "Stn Press (kPa)"
    ]).to_csv(weather_history_path, index=False)

    rows = []
    for fsa in fsas:
        for horizon in range(0, 25):
            rows.append({
                "timestamp": origin + pd.Timedelta(hours=horizon),
                "fsa": fsa,
                "Temp (°C)": pd.NA,
                "Rel Hum (%)": pd.NA,
                "Stn Press (kPa)": pd.NA,
            })
    pd.DataFrame(rows).to_csv(weather_forecast_path, index=False)

    return {
        "demand": demand_path,
        "weather_history": weather_history_path,
        "weather_forecast": weather_forecast_path,
    }
