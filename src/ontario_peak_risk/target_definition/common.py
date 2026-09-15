"""Shared utilities for Target Definition Phase."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_target_config(
    config_path: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Load the Target Definition configuration."""
    resolved = Path(config_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"Target Definition configuration not found: {resolved}"
        )

    with resolved.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The Target Definition configuration must be a YAML mapping."
        )

    return config, resolved.parent.parent


def resolve_project_path(
    project_root: Path,
    value: str,
) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def ensure_directories(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """Create output, report, and documentation directories."""
    output_dir = resolve_project_path(
        project_root,
        config["paths"]["output_dir"],
    )
    reports_dir = resolve_project_path(
        project_root,
        config["paths"]["reports_dir"],
    )
    docs_dir = resolve_project_path(
        project_root,
        config["paths"]["docs_dir"],
    )

    for path in (output_dir, reports_dir, docs_dir):
        path.mkdir(parents=True, exist_ok=True)

    return output_dir, reports_dir, docs_dir


def load_feature_dataset(
    config: dict[str, Any],
    project_root: Path,
) -> pd.DataFrame:
    """Load and order the Phase 6 feature dataset."""
    path = resolve_project_path(
        project_root,
        config["paths"]["input_feature_dataset"],
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}"
        )

    frame = pd.read_parquet(path)
    frame.columns = frame.columns.astype(str).str.strip()

    settings = config["target_definition"]
    required = set(
        settings["key_columns"]
        + [
            settings["forecasting"]["target_column"],
            "season",
            settings["peak_risk"]["diagnostic_time_column"],
        ]
    )

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Target Definition dataset is missing required columns: {missing}"
        )

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
    )

    return (
        frame
        .sort_values(
            ["fsa", "timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def save_table(
    frame: pd.DataFrame,
    path: str | Path,
) -> None:
    """Save a CSV report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )


def dataframe_to_markdown(
    frame: pd.DataFrame,
) -> str:
    """Convert a DataFrame to Markdown with a safe fallback."""
    if frame.empty:
        return "No records."

    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)
