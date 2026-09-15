"""Shared utilities for Phase 5 exploratory data analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_eda_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load EDA YAML and resolve the project root from configs/."""
    resolved = Path(config_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"EDA configuration not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The EDA configuration must contain a YAML mapping.")

    return config, resolved.parent.parent


def resolve_project_path(project_root: Path, value: str) -> Path:
    """Resolve a configuration path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def prepare_output_directories(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """Create and return reports, figures, and documentation directories."""
    reports = resolve_project_path(project_root, config["paths"]["reports_dir"])
    figures = resolve_project_path(project_root, config["paths"]["figures_dir"])
    docs = resolve_project_path(project_root, config["paths"]["docs_dir"])

    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    return reports, figures, docs


def load_clean_dataset(
    config: dict[str, Any],
    project_root: Path,
) -> pd.DataFrame:
    """Load the clean Master Dataset without modifying the source file."""
    path = resolve_project_path(
        project_root,
        config["paths"]["input_clean_dataset"],
    )

    if not path.exists():
        raise FileNotFoundError(f"Clean Master Dataset not found: {path}")

    frame = pd.read_parquet(path)
    frame.columns = frame.columns.astype(str).str.strip()

    required = set(config["columns"]["key"] + [config["columns"]["target"]])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"EDA dataset is missing required columns: {missing}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.sort_values(["fsa", "timestamp"], kind="stable").reset_index(drop=True)

    return frame


def save_table(
    frame: pd.DataFrame,
    path: str | Path,
) -> None:
    """Save a machine-readable table with consistent encoding."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")


def save_figure(
    figure: plt.Figure,
    path: str | Path,
) -> None:
    """Save and close a matplotlib figure."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Return a Markdown table with a safe fallback."""
    if frame.empty:
        return "No records."

    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric columns, excluding booleans if present."""
    return frame.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Return categorical and object-like columns."""
    return frame.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()


def correlation_strength(value: float) -> str:
    """Return a simple absolute-correlation interpretation."""
    absolute = abs(value)

    if absolute >= 0.70:
        return "strong"
    if absolute >= 0.40:
        return "moderate"
    if absolute >= 0.20:
        return "weak"
    return "very weak"


def write_section_markdown(
    path: Path,
    title: str,
    paragraphs: list[str],
    tables: list[tuple[str, pd.DataFrame]],
    figures: list[tuple[str, str]],
) -> None:
    """Write one EDA section report."""
    parts = [f"# {title}", ""]

    for paragraph in paragraphs:
        parts.extend([paragraph, ""])

    for heading, table in tables:
        parts.extend([f"## {heading}", "", dataframe_to_markdown(table), ""])

    for heading, relative_path in figures:
        parts.extend([f"## {heading}", "", f"![{heading}]({relative_path})", ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
