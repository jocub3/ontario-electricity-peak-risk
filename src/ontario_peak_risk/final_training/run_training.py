from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.ontario_peak_risk.modeling.common import (
    load_feature_dataset,
    load_forecast_targets,
    load_modeling_config,
)
from src.ontario_peak_risk.models.forecasting.random_forest import run_model as rfmod
from src.ontario_peak_risk.models.peak_risk.xgboost import run_model as xgbmod

def _yaml(path: Path):
    with path.open("r", encoding="utf-8") as h:
        return yaml.safe_load(h)

def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p

def run_final_training(config_path: str | Path = "configs/final_training.yaml") -> dict:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    cfg = _yaml(config_path)

    modeling_config, _ = load_modeling_config(
        _resolve(project_root, cfg["paths"]["modeling_foundation_config"])
    )
    rf_cfg = _yaml(_resolve(project_root, cfg["paths"]["random_forest_config"]))
    xgb_cfg = _yaml(_resolve(project_root, cfg["paths"]["xgboost_config"]))
    rf_freeze = _json(_resolve(project_root, cfg["paths"]["rf_freeze_manifest"]))
    xgb_freeze = _json(_resolve(project_root, cfg["paths"]["xgb_freeze_manifest"]))

    feature_dataset = load_feature_dataset(modeling_config, project_root)
    forecast_targets = load_forecast_targets(modeling_config, project_root).reset_index(drop=True)

    artifacts = _resolve(project_root, cfg["paths"]["artifacts_dir"])
    rf_dir = artifacts / "forecasting_random_forest"
    xgb_dir = artifacts / "peak_risk_xgboost"
    rf_dir.mkdir(parents=True, exist_ok=True)
    xgb_dir.mkdir(parents=True, exist_ok=True)

    # Only origins whose complete 24h target exists are already present in forecast_targets.
    rf_numeric, rf_categorical = rfmod.model_feature_columns(feature_dataset)

    for horizon in range(1, 25):
        target_col = f"{rf_cfg['model']['target_prefix']}{horizon:02d}"
        X = rfmod.build_horizon_features(
            forecast_targets[["fsa", "forecast_origin"]],
            feature_dataset,
            horizon=horizon,
        )[rf_numeric + rf_categorical]
        y = forecast_targets[target_col]

        model = rfmod._build_pipeline(
            rf_numeric,
            rf_categorical,
            rf_freeze["hyperparameters"],
            random_seed=int(rf_cfg["model"]["random_seed"]),
        )
        model.fit(X, y)
        joblib.dump(model, rf_dir / f"rf_h{horizon:02d}.joblib")
        print(f"[Final RF {horizon}/24] saved", flush=True)

    rf_meta = {
        "algorithm": "RandomForestRegressor",
        "horizons": 24,
        "hyperparameters": rf_freeze["hyperparameters"],
        "numeric_features": rf_numeric,
        "categorical_features": rf_categorical,
        "training_start": str(pd.to_datetime(forecast_targets["forecast_origin"]).min()),
        "training_end": str(pd.to_datetime(forecast_targets["forecast_origin"]).max()),
    }
    (rf_dir / "metadata.json").write_text(json.dumps(rf_meta, indent=2), encoding="utf-8")

    # XGBoost final training
    xgb_numeric, xgb_categorical = xgbmod.model_feature_columns(feature_dataset)
    peak_threshold_table = xgbmod.fit_peak_threshold_table(
        feature_dataset,
        percentile=float(xgb_cfg["model"]["peak_percentile"]),
    )
    peak_threshold_table.to_parquet(xgb_dir / "peak_thresholds.parquet", index=False)

    for horizon in range(1, 25):
        X, y, _ = xgbmod._prepare_horizon(
            forecast_targets,
            feature_dataset,
            peak_threshold_table,
            horizon,
            xgb_numeric,
            xgb_categorical,
        )
        class_ratio = xgbmod._class_ratio(y)
        model = xgbmod._build_pipeline(
            xgb_numeric,
            xgb_categorical,
            xgb_freeze["hyperparameters"],
            random_seed=int(xgb_cfg["model"]["random_seed"]),
            class_ratio=class_ratio,
        )
        model.fit(X, y)
        joblib.dump(model, xgb_dir / f"xgb_h{horizon:02d}.joblib")
        print(f"[Final XGB {horizon}/24] saved", flush=True)

    xgb_meta = {
        "algorithm": "XGBoostClassifier",
        "horizons": 24,
        "hyperparameters": xgb_freeze["hyperparameters"],
        "operational_threshold": float(xgb_freeze["operational_threshold"]),
        "peak_definition": xgb_freeze["peak_definition"],
        "numeric_features": xgb_numeric,
        "categorical_features": xgb_categorical,
        "training_start": str(pd.to_datetime(forecast_targets["forecast_origin"]).min()),
        "training_end": str(pd.to_datetime(forecast_targets["forecast_origin"]).max()),
    }
    (xgb_dir / "metadata.json").write_text(json.dumps(xgb_meta, indent=2), encoding="utf-8")

    manifest = {
        "status": "FINAL_MODELS_TRAINED",
        "rf_artifact_count": 24,
        "xgb_artifact_count": 24,
        "peak_threshold_table": "peak_risk_xgboost/peak_thresholds.parquet",
        "holdout_results_were_not_used_to_change_configuration": True,
    }
    (artifacts / "FINAL_MODEL_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return manifest
