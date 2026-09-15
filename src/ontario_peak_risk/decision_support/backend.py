"""Read-only backend used by the lightweight Streamlit application."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .common import resolve
from .contracts import validate_predictions


class DemoRepository:
    """Repository over precomputed artifacts; never loads model joblib files."""

    def __init__(self, cfg: dict, project_root: Path):
        self.cfg = cfg
        self.project_root = project_root
        self.threshold = float(cfg["application"]["official_peak_threshold"])
        self.fsas = list(cfg["application"]["fsas"])
        self.horizons = int(cfg["application"]["horizons"])

        prediction_path = resolve(project_root, cfg["paths"]["demo_predictions"])
        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Public demo predictions not found: {prediction_path}. "
                "Run DSA_02 first."
            )

        self.predictions = pd.read_parquet(prediction_path)
        self.predictions["forecast_origin"] = pd.to_datetime(
            self.predictions["forecast_origin"]
        )
        self.predictions["target_timestamp"] = pd.to_datetime(
            self.predictions["target_timestamp"]
        )

        report = validate_predictions(
            self.predictions, self.fsas, self.horizons, self.threshold
        )
        if report["status"].eq("FAIL").any():
            failed = report[report["status"].eq("FAIL")]
            raise ValueError(
                "Public demo prediction validation failed:\n"
                + failed.to_string(index=False)
            )

        scenario_path = resolve(
            project_root, cfg["paths"]["demo_weather_scenarios"]
        )
        self.scenarios = (
            pd.read_parquet(scenario_path) if scenario_path.exists()
            else pd.DataFrame()
        )
        if not self.scenarios.empty:
            self.scenarios["forecast_origin"] = pd.to_datetime(
                self.scenarios["forecast_origin"]
            )
            self.scenarios["target_timestamp"] = pd.to_datetime(
                self.scenarios["target_timestamp"]
            )

        metadata_path = resolve(project_root, cfg["paths"]["demo_metadata"])
        self.metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )

    @property
    def available_origins(self) -> list[pd.Timestamp]:
        return sorted(self.predictions["forecast_origin"].unique())

    def prediction_slice(
        self,
        origin: pd.Timestamp,
        fsas: list[str] | None = None,
    ) -> pd.DataFrame:
        fsas = fsas or self.fsas
        return (
            self.predictions[
                self.predictions["forecast_origin"].eq(pd.Timestamp(origin))
                & self.predictions["fsa"].isin(fsas)
            ]
            .sort_values(["fsa", "horizon"])
            .copy()
        )

    def scenario_slice(
        self,
        origin: pd.Timestamp,
        scenario: str,
        fsas: list[str] | None = None,
    ) -> pd.DataFrame:
        if self.scenarios.empty:
            return pd.DataFrame()
        fsas = fsas or self.fsas
        return (
            self.scenarios[
                self.scenarios["forecast_origin"].eq(pd.Timestamp(origin))
                & self.scenarios["scenario"].astype(str).eq(str(scenario))
                & self.scenarios["fsa"].isin(fsas)
            ]
            .sort_values(["fsa", "horizon"])
            .copy()
        )

    def executive_metrics(self, data: pd.DataFrame) -> dict:
        if data.empty:
            return {}

        combined = (
            data.groupby("target_timestamp", as_index=False)
            .agg(combined_demand_kwh=("forecast_consumption_kwh", "sum"))
            .sort_values("target_timestamp")
        )
        combined_row = combined.loc[combined["combined_demand_kwh"].idxmax()]
        demand_idx = data["forecast_consumption_kwh"].idxmax()
        risk_idx = data["peak_risk_score"].idxmax()
        demand_row = data.loc[demand_idx]
        risk_row = data.loc[risk_idx]
        alerts = data[data["peak_alert"].astype(bool)]

        return {
            "maximum_combined_demand_kwh": float(combined_row["combined_demand_kwh"]),
            "maximum_combined_demand_hour": pd.Timestamp(combined_row["target_timestamp"]),
            "maximum_fsa_demand_kwh": float(demand_row["forecast_consumption_kwh"]),
            "maximum_demand_fsa": str(demand_row["fsa"]),
            "maximum_fsa_demand_hour": pd.Timestamp(demand_row["target_timestamp"]),
            "highest_risk_score": float(risk_row["peak_risk_score"]),
            "highest_risk_fsa": str(risk_row["fsa"]),
            "highest_risk_hour": pd.Timestamp(risk_row["target_timestamp"]),
            "unique_alert_hours": int(alerts["target_timestamp"].nunique()),
            "fsa_hour_alerts": int(len(alerts)),
            "alert_fsas": int(alerts["fsa"].nunique()),
        }
