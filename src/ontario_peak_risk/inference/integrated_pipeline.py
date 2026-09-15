from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.ontario_peak_risk.models.forecasting.random_forest import run_model as rfmod
from src.ontario_peak_risk.models.peak_risk.xgboost import run_model as xgbmod

class IntegratedPredictionPipeline:
    """
    Load the frozen final Random Forest and XGBoost artifacts and return one
    24-hour prediction table for every requested FSA.

    IMPORTANT:
    `inference_feature_dataset` must already contain the same feature schema
    expected by the training pipeline, including forecast-available weather
    for target hours and leakage-safe historical-demand features.
    """

    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)

        self.rf_dir = self.artifacts_dir / "forecasting_random_forest"
        self.xgb_dir = self.artifacts_dir / "peak_risk_xgboost"

        self.rf_meta = json.loads((self.rf_dir / "metadata.json").read_text(encoding="utf-8"))
        self.xgb_meta = json.loads((self.xgb_dir / "metadata.json").read_text(encoding="utf-8"))

        self.peak_threshold = float(self.xgb_meta["operational_threshold"])

        self.rf_models = {
            h: joblib.load(self.rf_dir / f"rf_h{h:02d}.joblib")
            for h in range(1, 25)
        }
        self.xgb_models = {
            h: joblib.load(self.xgb_dir / f"xgb_h{h:02d}.joblib")
            for h in range(1, 25)
        }

    def predict(
        self,
        forecast_origin: str | pd.Timestamp,
        fsas: list[str],
        inference_feature_dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        origin = pd.Timestamp(forecast_origin)

        request = pd.DataFrame(
            {
                "fsa": fsas,
                "forecast_origin": [origin] * len(fsas),
            }
        )

        rows = []

        for horizon in range(1, 25):
            rf_features = rfmod.build_horizon_features(
                request,
                inference_feature_dataset,
                horizon=horizon,
            )
            xgb_features = xgbmod.build_horizon_features(
                request,
                inference_feature_dataset,
                horizon=horizon,
            )

            X_rf = rf_features[
                self.rf_meta["numeric_features"] + self.rf_meta["categorical_features"]
            ]
            X_xgb = xgb_features[
                self.xgb_meta["numeric_features"] + self.xgb_meta["categorical_features"]
            ]

            demand = self.rf_models[horizon].predict(X_rf)
            risk_probability = self.xgb_models[horizon].predict_proba(X_xgb)[:, 1]
            peak_alert = risk_probability >= self.peak_threshold

            target_timestamp = origin + pd.Timedelta(hours=horizon)

            rows.append(
                pd.DataFrame(
                    {
                        "fsa": fsas,
                        "forecast_origin": origin,
                        "target_timestamp": target_timestamp,
                        "horizon": horizon,
                        "forecast_consumption_kwh": demand,
                        "peak_risk_score": risk_probability,
                        "peak_alert": peak_alert.astype(bool),
                    }
                )
            )

        return pd.concat(rows, ignore_index=True).sort_values(
            ["fsa", "horizon"]
        ).reset_index(drop=True)

def save_prediction_outputs(frame: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "integrated_24h_prediction.csv", index=False)
    frame.to_parquet(output_dir / "integrated_24h_prediction.parquet", index=False)
