"""Pre-launch validation. Run after installing requirements."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml
from src.ontario_peak_risk.decision_support.backend import DemoRepository

cfg = yaml.safe_load((ROOT / "configs" / "decision_support_app.yaml").read_text(encoding="utf-8"))
repo = DemoRepository(cfg, ROOT)
assert len(repo.available_origins) >= 1
for origin in repo.available_origins:
    data = repo.prediction_slice(origin)
    assert len(data) == len(repo.fsas) * repo.horizons
    assert data["peak_risk_score"].between(0, 1).all()
    metrics = repo.executive_metrics(data)
    alerts = data[data["peak_alert"].astype(bool)]
    combined = data.groupby("target_timestamp")["forecast_consumption_kwh"].sum()
    assert metrics["maximum_combined_demand_kwh"] == float(combined.max())
    assert metrics["maximum_fsa_demand_kwh"] == float(data["forecast_consumption_kwh"].max())
    assert metrics["unique_alert_hours"] == int(alerts["target_timestamp"].nunique())
    assert metrics["fsa_hour_alerts"] == len(alerts)

    one_fsa = repo.prediction_slice(origin, [repo.fsas[0]])
    one_metrics = repo.executive_metrics(one_fsa)
    assert one_metrics["maximum_combined_demand_kwh"] == one_metrics["maximum_fsa_demand_kwh"]
assert not repo.scenarios.empty
print("Dashboard data contract: PASS")
print("Executive KPI definitions: PASS")
print("Origins:", len(repo.available_origins))
print("FSAs:", len(repo.fsas))
print("Horizons:", repo.horizons)
print("Prediction rows:", len(repo.predictions))
print("Scenario rows:", len(repo.scenarios))
