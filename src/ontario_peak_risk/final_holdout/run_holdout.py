from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import yaml

from src.ontario_peak_risk.modeling.common import (
    load_feature_dataset,
    load_forecast_targets,
    load_modeling_config,
)
from src.ontario_peak_risk.modeling.splits import (
    build_final_holdout_fold,
    slice_fold,
    slice_forecast_origin_fold,
)

from src.ontario_peak_risk.models.forecasting.random_forest import run_model as rfmod
from src.ontario_peak_risk.models.peak_risk.xgboost import run_model as xgbmod

from .metrics import (
    forecasting_metrics,
    classification_metrics,
    grouped_forecasting,
    grouped_classification,
)

def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p

def run_final_holdout(config_path: str | Path = "configs/final_holdout_2025.yaml") -> dict:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    cfg = _load_yaml(config_path)

    foundation_path = _resolve(project_root, cfg["paths"]["modeling_foundation_config"])
    modeling_config, _ = load_modeling_config(foundation_path)

    rf_cfg = _load_yaml(_resolve(project_root, cfg["paths"]["random_forest_config"]))
    xgb_cfg = _load_yaml(_resolve(project_root, cfg["paths"]["xgboost_config"]))

    rf_freeze = _load_json(_resolve(project_root, cfg["paths"]["rf_freeze_manifest"]))
    xgb_freeze = _load_json(_resolve(project_root, cfg["paths"]["xgb_freeze_manifest"]))

    if rf_freeze.get("final_2025_holdout_evaluated", False):
        raise ValueError("RF freeze manifest indicates the final holdout was already evaluated.")
    if xgb_freeze.get("final_2025_holdout_evaluated", False):
        raise ValueError("XGBoost freeze manifest indicates the final holdout was already evaluated.")

    feature_dataset = load_feature_dataset(modeling_config, project_root)
    forecast_targets = load_forecast_targets(modeling_config, project_root)

    holdout = build_final_holdout_fold(modeling_config)
    train_obs, test_obs = slice_fold(feature_dataset, holdout, time_column="timestamp")
    train_targets, test_targets = slice_forecast_origin_fold(
        forecast_targets,
        holdout,
        time_column="forecast_origin",
        max_horizon_hours=24,
    )
    train_targets = train_targets.reset_index(drop=True)
    test_targets = test_targets.reset_index(drop=True)

    reports_dir = _resolve(project_root, cfg["paths"]["reports_dir"])
    outputs_dir = _resolve(project_root, cfg["paths"]["outputs_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- RF Forecasting ----------------
    rf_numeric, rf_categorical = rfmod.model_feature_columns(feature_dataset)
    rf_parameters = rf_freeze["hyperparameters"]
    rf_parts = []
    started = time.perf_counter()

    for horizon in range(1, 25):
        target_column = f"{rf_cfg['model']['target_prefix']}{horizon:02d}"

        X_train = rfmod.build_horizon_features(
            train_targets[["fsa", "forecast_origin"]],
            feature_dataset,
            horizon=horizon,
        )[rf_numeric + rf_categorical]

        X_test = rfmod.build_horizon_features(
            test_targets[["fsa", "forecast_origin"]],
            feature_dataset,
            horizon=horizon,
        )[rf_numeric + rf_categorical]

        y_train = train_targets[target_column]
        y_test = test_targets[target_column]

        pipeline = rfmod._build_pipeline(
            rf_numeric,
            rf_categorical,
            rf_parameters,
            random_seed=int(rf_cfg["model"]["random_seed"]),
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        rf_parts.append(
            pd.DataFrame({
                "fsa": test_targets["fsa"].to_numpy(),
                "forecast_origin": test_targets["forecast_origin"].to_numpy(),
                "target_timestamp": (
                    pd.to_datetime(test_targets["forecast_origin"])
                    + pd.to_timedelta(horizon, unit="h")
                ),
                "horizon": horizon,
                "actual": y_test.to_numpy(),
                "predicted": pred,
            })
        )
        print(f"[RF Holdout {horizon}/24] completed", flush=True)

    rf_predictions = pd.concat(rf_parts, ignore_index=True)
    rf_predictions["month"] = pd.to_datetime(rf_predictions["target_timestamp"]).dt.month

    rf_global = pd.DataFrame([{
        "model": "RandomForestRegressor",
        **forecasting_metrics(rf_predictions["actual"], rf_predictions["predicted"]),
    }])
    rf_fsa = grouped_forecasting(rf_predictions, ["fsa"])
    rf_horizon = grouped_forecasting(rf_predictions, ["horizon"])
    rf_month = grouped_forecasting(rf_predictions, ["month"])

    # ---------------- XGB Peak-Risk ----------------
    xgb_numeric, xgb_categorical = xgbmod.model_feature_columns(feature_dataset)
    xgb_parameters = xgb_freeze["hyperparameters"]
    operational_threshold = float(cfg["peak_risk"]["operational_threshold"])

    manifest_threshold = float(xgb_freeze["operational_threshold"])
    if abs(operational_threshold - manifest_threshold) > 1e-12:
        raise ValueError(
            f"Configured threshold {operational_threshold} differs from frozen threshold {manifest_threshold}."
        )

    threshold_table = xgbmod.fit_peak_threshold_table(
        train_obs,
        percentile=float(xgb_cfg["model"]["peak_percentile"]),
    )

    xgb_parts = []

    for horizon in range(1, 25):
        X_train, y_train, _ = xgbmod._prepare_horizon(
            train_targets,
            feature_dataset,
            threshold_table,
            horizon,
            xgb_numeric,
            xgb_categorical,
        )
        X_test, y_test, keys = xgbmod._prepare_horizon(
            test_targets,
            feature_dataset,
            threshold_table,
            horizon,
            xgb_numeric,
            xgb_categorical,
        )

        class_ratio = xgbmod._class_ratio(y_train)
        pipeline = xgbmod._build_pipeline(
            xgb_numeric,
            xgb_categorical,
            xgb_parameters,
            random_seed=int(xgb_cfg["model"]["random_seed"]),
            class_ratio=class_ratio,
        )
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_test)[:, 1]
        predicted = (probability >= operational_threshold).astype("int8")

        xgb_parts.append(
            pd.DataFrame({
                "fsa": keys["fsa"].to_numpy(),
                "forecast_origin": keys["forecast_origin"].to_numpy(),
                "target_timestamp": keys["target_timestamp"].to_numpy(),
                "season": keys["season"].to_numpy(),
                "peak_threshold": keys["peak_threshold"].to_numpy(),
                "horizon": horizon,
                "actual": y_test.to_numpy(),
                "probability": probability,
                "predicted": predicted,
            })
        )
        print(f"[XGB Holdout {horizon}/24] completed", flush=True)

    xgb_predictions = pd.concat(xgb_parts, ignore_index=True)
    xgb_predictions["month"] = pd.to_datetime(xgb_predictions["target_timestamp"]).dt.month

    xgb_global = pd.DataFrame([{
        "model": "XGBoostClassifier",
        "operational_threshold": operational_threshold,
        **classification_metrics(
            xgb_predictions["actual"],
            xgb_predictions["predicted"],
            xgb_predictions["probability"],
        ),
    }])
    xgb_fsa = grouped_classification(xgb_predictions, ["fsa"])
    xgb_horizon = grouped_classification(xgb_predictions, ["horizon"])
    xgb_month = grouped_classification(xgb_predictions, ["month"])

    # Save
    tables = {
        "FH_02_rf_global_metrics.csv": rf_global,
        "FH_02_rf_fsa_metrics.csv": rf_fsa,
        "FH_02_rf_horizon_metrics.csv": rf_horizon,
        "FH_02_rf_month_metrics.csv": rf_month,
        "FH_03_xgb_global_metrics.csv": xgb_global,
        "FH_03_xgb_fsa_metrics.csv": xgb_fsa,
        "FH_03_xgb_horizon_metrics.csv": xgb_horizon,
        "FH_03_xgb_month_metrics.csv": xgb_month,
    }
    for name, frame in tables.items():
        frame.to_csv(reports_dir / name, index=False)

    rf_predictions.to_parquet(outputs_dir / "rf_2025_holdout_predictions.parquet", index=False)
    xgb_predictions.to_parquet(outputs_dir / "xgb_2025_holdout_predictions.parquet", index=False)

    summary = {
        "holdout": holdout.name,
        "test_start": str(holdout.evaluation_start),
        "test_end": str(holdout.evaluation_end),
        "rf_frozen_parameters": rf_parameters,
        "xgb_frozen_parameters": xgb_parameters,
        "xgb_operational_threshold": operational_threshold,
        "brier_score": float(xgb_global.iloc[0]["brier_score"]),
        "parameter_changes_after_holdout_allowed": False,
    }
    (outputs_dir / "FINAL_HOLDOUT_2025_MANIFEST.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("FINAL HOLDOUT 2025 COMPLETED")
    print("No post-holdout tuning is permitted.")
    print("=" * 72)

    return {
        "rf_predictions": rf_predictions,
        "rf_global": rf_global,
        "rf_fsa": rf_fsa,
        "rf_horizon": rf_horizon,
        "rf_month": rf_month,
        "xgb_predictions": xgb_predictions,
        "xgb_global": xgb_global,
        "xgb_fsa": xgb_fsa,
        "xgb_horizon": xgb_horizon,
        "xgb_month": xgb_month,
    }
