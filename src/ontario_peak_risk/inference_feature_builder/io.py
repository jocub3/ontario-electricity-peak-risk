"""Incremental operational-data loading for inference."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

from src.ontario_peak_risk.modeling.common import (
    load_feature_dataset,
    load_modeling_config,
)


def load_ifb_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, path.parent.parent


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def normalize_timestamp(series: pd.Series, timezone: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(timezone).dt.tz_localize(None)
    return parsed


def _read_csv_files(
    directory: Path,
    pattern: str,
    *,
    timezone: str,
    timestamp_column: str,
    source_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(directory.glob(pattern)) if directory.exists() else []
    frames, inventory = [], []

    for order, path in enumerate(files):
        frame = pd.read_csv(path)
        frame.columns = frame.columns.astype(str).str.strip()
        if timestamp_column in frame.columns:
            frame[timestamp_column] = normalize_timestamp(
                frame[timestamp_column], timezone
            )

        frame["_operational_source_file"] = path.name
        frame["_operational_source_order"] = order
        frame["_operational_source_type"] = source_type
        frames.append(frame)

        inventory.append(
            {
                "source_type": source_type,
                "file": path.name,
                "path": str(path),
                "rows": len(frame),
                "min_timestamp": (
                    frame[timestamp_column].min()
                    if timestamp_column in frame.columns else pd.NaT
                ),
                "max_timestamp": (
                    frame[timestamp_column].max()
                    if timestamp_column in frame.columns else pd.NaT
                ),
            }
        )

    combined = (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames else pd.DataFrame()
    )
    return combined, pd.DataFrame(inventory)


def load_project_history(config: dict, project_root: Path) -> pd.DataFrame:
    modeling_path = resolve_path(
        project_root, config["paths"]["modeling_foundation_config"]
    )
    modeling_config, _ = load_modeling_config(modeling_path)
    feature_dataset = load_feature_dataset(modeling_config, project_root)

    cols = config["columns"]
    candidates = [
        cols["timestamp"],
        cols["fsa"],
        cols["demand"],
        cols["premise_count"],
        cols["temperature"],
        cols["humidity"],
        cols["pressure"],
    ]
    existing = [c for c in candidates if c in feature_dataset.columns]

    history = feature_dataset[existing].copy()
    history["_operational_source_file"] = "PROJECT_HISTORY"
    history["_operational_source_order"] = -1
    history["_operational_source_type"] = "project_history"
    return history


def load_operational_updates(
    config: dict,
    project_root: Path,
) -> dict[str, pd.DataFrame]:
    timestamp = config["columns"]["timestamp"]
    timezone = config["inference"]["timezone"]

    demand, demand_inventory = _read_csv_files(
        resolve_path(project_root, config["paths"]["demand_updates_dir"]),
        config["operational_files"]["demand_pattern"],
        timezone=timezone,
        timestamp_column=timestamp,
        source_type="demand_update",
    )
    weather_history, history_inventory = _read_csv_files(
        resolve_path(project_root, config["paths"]["weather_history_updates_dir"]),
        config["operational_files"]["weather_history_pattern"],
        timezone=timezone,
        timestamp_column=timestamp,
        source_type="weather_history_update",
    )
    weather_forecasts, forecast_inventory = _read_csv_files(
        resolve_path(project_root, config["paths"]["weather_forecasts_dir"]),
        config["operational_files"]["weather_forecast_pattern"],
        timezone=timezone,
        timestamp_column=timestamp,
        source_type="weather_forecast",
    )

    inventory = pd.concat(
        [demand_inventory, history_inventory, forecast_inventory],
        ignore_index=True,
        sort=False,
    )
    return {
        "demand": demand,
        "weather_history": weather_history,
        "weather_forecasts": weather_forecasts,
        "inventory": inventory,
    }


def _latest_by_key(frame: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    order_cols = [*key, "_operational_source_order"]
    return (
        frame.sort_values(order_cols, kind="stable")
        .drop_duplicates(key, keep="last")
        .reset_index(drop=True)
    )


def combine_observed_history(
    project_history: pd.DataFrame,
    demand_updates: pd.DataFrame,
    weather_history_updates: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = config["columns"]
    fsa, timestamp = cols["fsa"], cols["timestamp"]
    key = [fsa, timestamp]

    base_cols = [
        fsa, timestamp, cols["demand"], cols["premise_count"],
        cols["temperature"], cols["humidity"], cols["pressure"],
    ]
    base_cols = [c for c in base_cols if c in project_history.columns]
    combined = project_history[base_cols].copy()

    if not demand_updates.empty:
        keep = [
            c for c in [
                fsa, timestamp, cols["demand"], cols["premise_count"],
                "_operational_source_order"
            ]
            if c in demand_updates.columns
        ]
        demand = _latest_by_key(demand_updates[keep].copy(), key)
        demand_values = [c for c in [cols["demand"], cols["premise_count"]]
                         if c in demand.columns]
        combined = combined.merge(
            demand[key + demand_values],
            on=key,
            how="outer",
            suffixes=("", "__update"),
        )
        for col in demand_values:
            update = f"{col}__update"
            if update in combined.columns:
                combined[col] = combined[update].combine_first(combined.get(col))
                combined = combined.drop(columns=update)

    if not weather_history_updates.empty:
        keep = [
            c for c in [
                fsa, timestamp, cols["temperature"], cols["humidity"],
                cols["pressure"], "_operational_source_order"
            ]
            if c in weather_history_updates.columns
        ]
        weather = _latest_by_key(weather_history_updates[keep].copy(), key)
        weather_values = [
            c for c in [cols["temperature"], cols["humidity"], cols["pressure"]]
            if c in weather.columns
        ]
        combined = combined.merge(
            weather[key + weather_values],
            on=key,
            how="outer",
            suffixes=("", "__update"),
        )
        for col in weather_values:
            update = f"{col}__update"
            if update in combined.columns:
                combined[col] = combined[update].combine_first(combined.get(col))
                combined = combined.drop(columns=update)

    combined[timestamp] = normalize_timestamp(
        combined[timestamp], config["inference"]["timezone"]
    )
    combined[fsa] = combined[fsa].astype(str).str.strip()

    duplicate_report = (
        combined.loc[combined.duplicated(key, keep=False), key]
        .value_counts()
        .rename("duplicate_count")
        .reset_index()
    )

    combined = (
        combined.sort_values(key, kind="stable")
        .drop_duplicates(key, keep=config["duplicate_policy"]["keep"])
        .reset_index(drop=True)
    )
    return combined, duplicate_report


def latest_weather_forecast_for_origin(
    weather_forecasts: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    fsas: list[str],
    config: dict,
) -> pd.DataFrame:
    if weather_forecasts.empty:
        return weather_forecasts.copy()

    cols = config["columns"]
    key = [cols["fsa"], cols["timestamp"]]
    frame = weather_forecasts.copy()
    frame[cols["fsa"]] = frame[cols["fsa"]].astype(str).str.strip()

    end = forecast_origin + pd.Timedelta(
        hours=int(config["inference"]["horizons"])
    )
    frame = frame.loc[
        frame[cols["fsa"]].isin(fsas)
        & frame[cols["timestamp"]].between(
            forecast_origin, end, inclusive="both"
        )
    ].copy()

    return _latest_by_key(frame, key).sort_values(key).reset_index(drop=True)


def observed_history_for_origin(
    project_history: pd.DataFrame,
    demand_updates: pd.DataFrame,
    weather_history_updates: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    config: dict,
    *,
    extra_buffer_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build only the observed-history window required for inference.

    This avoids merging the entire 2021-2025 history with all operational
    updates for every prediction request. The maximum configured lookback is
    retained, plus an optional buffer.

    No post-origin demand is included in the returned inference history.
    """
    origin = pd.Timestamp(forecast_origin)
    ts = config["columns"]["timestamp"]

    lookback = int(
        config["inference"]["required_history_hours_before_origin"]
    )
    start = origin - pd.Timedelta(
        hours=lookback + int(extra_buffer_hours)
    )

    def slice_window(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or ts not in frame.columns:
            return frame.copy()
        return frame.loc[
            pd.to_datetime(frame[ts]).between(
                start,
                origin,
                inclusive="both",
            )
        ].copy()

    return combine_observed_history(
        slice_window(project_history),
        slice_window(demand_updates),
        slice_window(weather_history_updates),
        config,
    )
