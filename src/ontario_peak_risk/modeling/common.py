"""Shared utilities for Modeling Foundation Phase."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_modeling_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load the Modeling Foundation YAML configuration."""
    resolved = Path(config_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Modeling Foundation configuration not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The Modeling Foundation configuration must be a YAML mapping.")

    return config, resolved.parent.parent


def resolve_project_path(project_root: Path, value: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def ensure_modeling_directories(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """Create reports, documentation, and output directories."""
    reports_dir = resolve_project_path(project_root, config["paths"]["reports_dir"])
    docs_dir = resolve_project_path(project_root, config["paths"]["docs_dir"])
    outputs_dir = resolve_project_path(project_root, config["paths"]["outputs_dir"])

    for path in (reports_dir, docs_dir, outputs_dir):
        path.mkdir(parents=True, exist_ok=True)

    return reports_dir, docs_dir, outputs_dir


def load_feature_dataset(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Load the Phase 6 feature dataset."""
    path = resolve_project_path(project_root, config["paths"]["feature_dataset"])
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")

    frame = pd.read_parquet(path)
    frame.columns = frame.columns.astype(str).str.strip()

    time_column = config["modeling"]["time"]["time_column"]
    group_column = config["modeling"]["time"]["group_column"]

    required = {time_column, group_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {missing}")

    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")

    return (
        frame.sort_values([group_column, time_column], kind="stable")
        .reset_index(drop=True)
    )


def load_forecast_targets(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Load the complete 24-hour Forecasting target matrix."""
    path = resolve_project_path(project_root, config["paths"]["forecast_targets"])
    if not path.exists():
        raise FileNotFoundError(f"Forecasting target dataset not found: {path}")

    frame = pd.read_parquet(path)
    if "forecast_origin" not in frame.columns:
        raise ValueError("Forecast target dataset must contain 'forecast_origin'.")

    frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"], errors="coerce")

    return (
        frame.sort_values(["fsa", "forecast_origin"], kind="stable")
        .reset_index(drop=True)
    )


def save_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Save a CSV report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Convert a DataFrame to Markdown with a fallback."""
    if frame.empty:
        return "No records."
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)
