"""Discover valid forecast origins from actual operational files."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

from .common import resolve


def _read_csvs(directory: Path) -> pd.DataFrame:
    files = sorted(p for p in directory.glob("*.csv") if p.name != ".gitkeep")
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    frames = []
    for file in files:
        frame = pd.read_csv(file)
        frame["__source_file"] = file.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _normalize_timestamp(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise KeyError(f"Expected timestamp column '{column}' not found.")
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    if out[column].isna().any():
        bad = int(out[column].isna().sum())
        raise ValueError(f"{bad} timestamp values could not be parsed in '{column}'.")
    return out


def discover_available_origins(
    cfg: dict,
    project_root: Path,
) -> pd.DataFrame:
    """
    Find forecast origins supported by actual recent demand and weather files.

    The method does not invent dates. An origin is accepted only when every
    configured FSA has:
      * observed demand from origin-168h through origin, and
      * weather rows from origin through origin+24h.
    """
    ifb_path = resolve(project_root, cfg["paths"]["ifb_config"])
    with ifb_path.open("r", encoding="utf-8") as handle:
        ifb = yaml.safe_load(handle)

    timestamp = ifb["columns"]["timestamp"]
    fsa_col = ifb["columns"]["fsa"]

    demand = _normalize_timestamp(
        _read_csvs(resolve(project_root, cfg["paths"]["operational_demand_dir"])),
        timestamp,
    )
    weather = _normalize_timestamp(
        _read_csvs(
            resolve(project_root, cfg["paths"]["operational_weather_forecast_dir"])
        ),
        timestamp,
    )

    demand[fsa_col] = demand[fsa_col].astype(str).str.upper().str.strip()
    weather[fsa_col] = weather[fsa_col].astype(str).str.upper().str.strip()

    fsas = list(cfg["application"]["fsas"])
    history_h = int(cfg["demo"]["required_history_hours"])
    weather_h = int(cfg["demo"]["required_weather_hours_ahead"])

    demand = demand[demand[fsa_col].isin(fsas)].drop_duplicates([fsa_col, timestamp])
    weather = weather[weather[fsa_col].isin(fsas)].drop_duplicates([fsa_col, timestamp])

    # Candidate origins must exist for all FSAs in both data sources.
    demand_counts = demand.groupby(timestamp)[fsa_col].nunique()
    weather_counts = weather.groupby(timestamp)[fsa_col].nunique()
    candidates = sorted(
        set(demand_counts[demand_counts.eq(len(fsas))].index)
        & set(weather_counts[weather_counts.eq(len(fsas))].index)
    )

    demand_keys = set(zip(demand[fsa_col], demand[timestamp]))
    weather_keys = set(zip(weather[fsa_col], weather[timestamp]))

    rows = []
    for origin in candidates:
        required_demand = pd.date_range(
            origin - pd.Timedelta(hours=history_h),
            origin,
            freq="h",
        )
        required_weather = pd.date_range(
            origin,
            origin + pd.Timedelta(hours=weather_h),
            freq="h",
        )

        demand_ok = all(
            (fsa, ts) in demand_keys for fsa in fsas for ts in required_demand
        )
        weather_ok = all(
            (fsa, ts) in weather_keys for fsa in fsas for ts in required_weather
        )

        if demand_ok and weather_ok:
            rows.append(
                {
                    "forecast_origin": origin,
                    "demand_history_status": "PASS",
                    "weather_grid_status": "PASS",
                    "fsa_count": len(fsas),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "forecast_origin",
                "demand_history_status",
                "weather_grid_status",
                "fsa_count",
            ]
        )

    return result.sort_values("forecast_origin").reset_index(drop=True)


def choose_demo_origins(
    available: pd.DataFrame,
    max_origins: int,
    strategy: str = "spread",
) -> pd.DataFrame:
    """Choose representative origins only from the validated available-origin table."""
    if available.empty:
        raise ValueError("No valid forecast origins are available.")

    data = available.sort_values("forecast_origin").reset_index(drop=True)
    n = min(int(max_origins), len(data))

    if n == len(data):
        return data.copy()

    if strategy == "spread":
        import numpy as np

        positions = np.linspace(0, len(data) - 1, n).round().astype(int)
        positions = sorted(set(int(i) for i in positions))
        return data.iloc[positions].reset_index(drop=True)

    if strategy == "latest":
        return data.tail(n).reset_index(drop=True)

    raise ValueError(f"Unsupported demo-origin selection strategy: {strategy}")
