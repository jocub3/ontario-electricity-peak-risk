"""Shared configuration, paths, metadata, and feature-family utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def load_explainability_config(
    config_path: str | Path = "configs/explainability.yaml",
) -> tuple[dict, Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, path.parent.parent


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def ensure_phase_directories(config: dict, project_root: Path) -> dict[str, Path]:
    paths = {}
    for key in ["reports_dir", "figures_dir", "outputs_dir", "docs_dir"]:
        path = resolve_path(project_root, config["paths"][key])
        path.mkdir(parents=True, exist_ok=True)
        paths[key] = path
    return paths


def load_final_metadata(config: dict, project_root: Path) -> dict[str, dict]:
    artifacts = resolve_path(project_root, config["paths"]["artifacts_dir"])
    rf = json.loads(
        (artifacts / config["models"]["rf_subdir"] / "metadata.json")
        .read_text(encoding="utf-8")
    )
    xgb = json.loads(
        (artifacts / config["models"]["xgb_subdir"] / "metadata.json")
        .read_text(encoding="utf-8")
    )
    return {"rf": rf, "xgb": xgb}


def public_features(metadata: dict) -> list[str]:
    return (
        list(metadata["numeric_features"])
        + list(metadata["categorical_features"])
    )


def feature_family(feature: str, config: dict) -> str:
    for family, members in config["feature_families"].items():
        if feature in members:
            return family
    return "other"


def add_feature_family(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = frame.copy()
    result["feature_family"] = result["feature"].map(
        lambda value: feature_family(str(value), config)
    )
    return result


def aggregate_by_family(
    frame: pd.DataFrame,
    value_column: str,
    config: dict,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    work = add_feature_family(frame, config)
    groups = list(group_columns or []) + ["feature_family"]
    return (
        work.groupby(groups, observed=True)[value_column]
        .sum()
        .reset_index()
        .sort_values(groups[:-1] + [value_column], ascending=False)
        .reset_index(drop=True)
    )
