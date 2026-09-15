"""Freeze the final Random Forest forecasting development configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    *,
    selected_parameters: dict,
    selection_evidence: dict,
    original_config_path: Path,
    refinement_config_path: Path,
    freeze_config: dict,
) -> dict:
    return {
        "status": "FROZEN_FOR_FINAL_HOLDOUT",
        "task": "forecasting",
        "algorithm": freeze_config["algorithm"],
        "forecast_horizons": int(freeze_config["horizons"]),
        "hyperparameters": selected_parameters,
        "feature_policy": freeze_config["feature_policy"],
        "preprocessing_policy": freeze_config["preprocessing_policy"],
        "selection_evidence": selection_evidence,
        "original_model_config_sha256": sha256(original_config_path),
        "refinement_config_sha256": sha256(refinement_config_path),
        "final_2025_holdout_evaluated": False,
        "holdout_policy": freeze_config["holdout_policy"],
    }


def save_manifest(manifest: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return path
