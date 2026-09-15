"""Freeze the selected XGBoost Peak-Risk configuration before 2025 holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_freeze_manifest(
    *,
    project_root: Path,
    config: dict,
    selected_threshold: float,
    threshold_evidence: dict,
) -> dict:
    paths = config["paths"]

    best_parameters_path = (
        project_root / paths["best_parameters"]
    )
    feature_dictionary_path = (
        project_root / paths["feature_dictionary"]
    )
    predictions_path = (
        project_root / paths["validation_predictions"]
    )

    if best_parameters_path.exists():
        best_parameters = json.loads(
            best_parameters_path.read_text(encoding="utf-8")
        )
    else:
        best_parameters = None

    if feature_dictionary_path.exists():
        feature_dictionary = pd.read_csv(
            feature_dictionary_path
        )
        frozen_features = (
            feature_dictionary["feature"].astype(str).tolist()
            if "feature" in feature_dictionary.columns
            else feature_dictionary.to_dict(orient="records")
        )
    else:
        frozen_features = None

    return {
        "status": "FROZEN_FOR_FINAL_HOLDOUT",
        "algorithm": "XGBoostClassifier",
        "operational_threshold": float(selected_threshold),
        "peak_definition": config["freeze"]["peak_definition"],
        "forecast_horizons": int(config["freeze"]["horizons"]),
        "hyperparameters": best_parameters,
        "features": frozen_features,
        "preprocessing_policy": config["freeze"]["preprocessing_policy"],
        "hyperparameter_policy": config["freeze"]["hyperparameter_policy"],
        "feature_policy": config["freeze"]["feature_policy"],
        "threshold_policy": config["freeze"]["threshold_policy"],
        "holdout_policy": config["freeze"]["holdout_policy"],
        "threshold_selection_evidence": threshold_evidence,
        "validation_predictions_sha256": file_sha256(
            predictions_path
        ),
        "best_parameters_sha256": file_sha256(
            best_parameters_path
        ),
        "feature_dictionary_sha256": file_sha256(
            feature_dictionary_path
        ),
        "final_2025_holdout_evaluated": False,
    }


def save_freeze_manifest(
    manifest: dict,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return output_path
