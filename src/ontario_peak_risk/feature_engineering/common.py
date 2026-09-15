"""Shared utilities for Phase 6 Feature Engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_feature_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load the Feature Engineering YAML configuration."""
    resolved = Path(config_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Feature Engineering configuration not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The Feature Engineering configuration must be a YAML mapping.")

    return config, resolved.parent.parent


def resolve_project_path(project_root: Path, value: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def ensure_directories(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[Path, Path]:
    """Create report and documentation directories."""
    reports = resolve_project_path(project_root, config["paths"]["reports_dir"])
    docs = resolve_project_path(project_root, config["paths"]["docs_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    return reports, docs


def load_clean_dataset(
    config: dict[str, Any],
    project_root: Path,
) -> pd.DataFrame:
    """Load the clean Master Dataset and enforce deterministic ordering."""
    path = resolve_project_path(
        project_root,
        config["paths"]["input_clean_dataset"],
    )
    if not path.exists():
        raise FileNotFoundError(f"Clean Master Dataset not found: {path}")

    frame = pd.read_parquet(path)
    frame.columns = frame.columns.astype(str).str.strip()

    required = set(
        config["feature_engineering"]["key_columns"]
        + [config["feature_engineering"]["target_column"]]
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return (
        frame
        .sort_values(["fsa", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )


def save_table(frame: pd.DataFrame, path: str | Path) -> None:
    """Save a CSV report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Convert a DataFrame to Markdown with a safe fallback."""
    if frame.empty:
        return "No records."
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)


def make_feature_log(
    feature_name: str,
    feature_family: str,
    source_columns: list[str],
    description: str,
    applies_to_forecasting: bool,
    applies_to_peak_risk: bool,
    computation_scope: str,
    leakage_risk: str = "low",
) -> dict[str, object]:
    """Create one standardized feature-dictionary record."""
    return {
        "feature_name": feature_name,
        "feature_family": feature_family,
        "source_columns": " | ".join(source_columns),
        "description": description,
        "forecasting_candidate": applies_to_forecasting,
        "peak_risk_candidate": applies_to_peak_risk,
        "computation_scope": computation_scope,
        "leakage_risk": leakage_risk,
    }
