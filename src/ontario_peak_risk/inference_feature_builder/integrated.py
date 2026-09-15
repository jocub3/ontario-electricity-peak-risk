"""Execute final serialized models from IFB-built feature frames."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd


class OperationalIntegratedPipeline:
    """
    Operational wrapper around the 24 RF + 24 XGBoost frozen model artifacts.

    Important:
    - Model artifacts are cached at process level after the first load.
    - The official operational threshold is read from XGBoost metadata.
    - `decision_threshold` may be overridden for scenario/sensitivity analysis
      without retraining the classifier. Such an override changes only the
      alerting decision rule, not the predicted risk score.
    """

    _BUNDLE_CACHE: dict[str, dict] = {}

    def __init__(
        self,
        artifacts_dir: str | Path,
        *,
        decision_threshold: float | None = None,
        load_workers: int = 4,
        use_cache: bool = True,
    ):
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.rf_dir = self.artifacts_dir / "forecasting_random_forest"
        self.xgb_dir = self.artifacts_dir / "peak_risk_xgboost"

        cache_key = str(self.artifacts_dir)
        started = perf_counter()

        if use_cache and cache_key in self._BUNDLE_CACHE:
            bundle = self._BUNDLE_CACHE[cache_key]
            self.loaded_from_cache = True
        else:
            bundle = self._load_bundle(max(1, int(load_workers)))
            if use_cache:
                self._BUNDLE_CACHE[cache_key] = bundle
            self.loaded_from_cache = False

        self.model_load_seconds = perf_counter() - started

        self.rf_meta = bundle["rf_meta"]
        self.xgb_meta = bundle["xgb_meta"]
        self.rf_models = bundle["rf_models"]
        self.xgb_models = bundle["xgb_models"]

        self.official_threshold = float(
            self.xgb_meta["operational_threshold"]
        )
        self.threshold = self._resolve_threshold(decision_threshold)

    def _resolve_threshold(self, value: float | None) -> float:
        threshold = self.official_threshold if value is None else float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "decision_threshold must be between 0 and 1."
            )
        return threshold

    def set_decision_threshold(self, value: float | None = None) -> None:
        """
        Change only the alert decision threshold.

        Passing None restores the official frozen threshold.
        No model is retrained and no risk score changes.
        """
        self.threshold = self._resolve_threshold(value)

    @classmethod
    def clear_model_cache(cls) -> None:
        """Clear cached model artifacts for the current Python process."""
        cls._BUNDLE_CACHE.clear()

    def _load_bundle(self, workers: int) -> dict:
        rf_meta = json.loads(
            (self.rf_dir / "metadata.json").read_text(encoding="utf-8")
        )
        xgb_meta = json.loads(
            (self.xgb_dir / "metadata.json").read_text(encoding="utf-8")
        )

        def load_rf(horizon: int):
            return (
                horizon,
                joblib.load(self.rf_dir / f"rf_h{horizon:02d}.joblib"),
            )

        def load_xgb(horizon: int):
            return (
                horizon,
                joblib.load(self.xgb_dir / f"xgb_h{horizon:02d}.joblib"),
            )

        horizons = list(range(1, 25))

        if workers == 1:
            rf_models = dict(load_rf(h) for h in horizons)
            xgb_models = dict(load_xgb(h) for h in horizons)
        else:
            # Loading is largely disk/deserialization work. A small thread
            # pool can reduce cold-start time on many local systems without
            # spawning multiple Python processes or duplicating memory.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rf_models = dict(executor.map(load_rf, horizons))
                xgb_models = dict(executor.map(load_xgb, horizons))

        return {
            "rf_meta": rf_meta,
            "xgb_meta": xgb_meta,
            "rf_models": rf_models,
            "xgb_models": xgb_models,
        }

    def predict_from_feature_frames(
        self,
        feature_frames: dict[str, dict[int, pd.DataFrame]],
    ) -> pd.DataFrame:
        rf_features = (
            list(self.rf_meta["numeric_features"])
            + list(self.rf_meta["categorical_features"])
        )
        xgb_features = (
            list(self.xgb_meta["numeric_features"])
            + list(self.xgb_meta["categorical_features"])
        )

        rows = []

        for horizon in range(1, 25):
            rf_frame = feature_frames["rf"][horizon].copy()
            xgb_frame = feature_frames["xgb"][horizon].copy()

            keys = [
                "fsa",
                "forecast_origin",
                "target_timestamp",
                "horizon",
            ]
            if not rf_frame[keys].equals(xgb_frame[keys]):
                raise ValueError(
                    f"RF/XGBoost inference keys do not align at h+{horizon}."
                )

            demand = self.rf_models[horizon].predict(
                rf_frame[rf_features]
            )
            probability = self.xgb_models[horizon].predict_proba(
                xgb_frame[xgb_features]
            )[:, 1]

            rows.append(
                pd.DataFrame(
                    {
                        "fsa": rf_frame["fsa"].to_numpy(),
                        "forecast_origin":
                            rf_frame["forecast_origin"].to_numpy(),
                        "target_timestamp":
                            rf_frame["target_timestamp"].to_numpy(),
                        "horizon": horizon,
                        "forecast_consumption_kwh": demand,
                        "peak_risk_score": probability,
                        "peak_alert": probability >= self.threshold,
                    }
                )
            )

        return (
            pd.concat(rows, ignore_index=True)
            .sort_values(["fsa", "horizon"])
            .reset_index(drop=True)
        )
