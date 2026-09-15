"""Memory-conscious access to final horizon-specific model artifacts."""

from __future__ import annotations

import gc
from pathlib import Path

import joblib


def model_path(
    task: str,
    horizon: int,
    config: dict,
    project_root: Path,
) -> Path:
    artifacts = project_root / config["paths"]["artifacts_dir"]

    if task == "rf":
        directory = artifacts / config["models"]["rf_subdir"]
        filename = config["models"]["rf_pattern"].format(
            horizon=int(horizon)
        )
    elif task == "xgb":
        directory = artifacts / config["models"]["xgb_subdir"]
        filename = config["models"]["xgb_pattern"].format(
            horizon=int(horizon)
        )
    else:
        raise ValueError("task must be 'rf' or 'xgb'.")

    return directory / filename


def load_horizon_pipeline(
    task: str,
    horizon: int,
    config: dict,
    project_root: Path,
):
    path = model_path(task, horizon, config, project_root)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def release_model(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
