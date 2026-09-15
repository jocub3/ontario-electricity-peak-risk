"""Functional checks for the redesigned Streamlit decision-support app."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

import yaml
from streamlit.testing.v1 import AppTest

from dashboard.dsa_app import charts
from src.ontario_peak_risk.decision_support.backend import DemoRepository


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "decision_support_app.yaml").read_text(encoding="utf-8"))
    repo = DemoRepository(cfg, ROOT)
    origin = repo.available_origins[-1]
    data = repo.prediction_slice(origin)
    metrics = repo.executive_metrics(data)

    assert metrics["maximum_combined_demand_kwh"] >= metrics["maximum_fsa_demand_kwh"]
    assert metrics["fsa_hour_alerts"] >= metrics["unique_alert_hours"]

    combined_figure = charts.combined_forecast_line(data)
    decision_figure = charts.demand_decision_heatmap(data, repo.threshold)
    figures = [
        combined_figure,
        decision_figure,
        charts.demand_heatmap(data),
        charts.forecast_lines(data),
        charts.risk_timeline(data, repo.threshold),
    ]
    for figure in figures:
        assert figure.data
        figure.to_json()
    assert any(trace.name == "Peak-Risk alert hour" for trace in combined_figure.data)
    assert decision_figure.data[0].colorbar.title.text == "Demand (kWh)"

    scenarios = repo.scenarios[repo.scenarios["forecast_origin"].eq(origin)]
    new_alert_rows = scenarios[
        scenarios["peak_alert"].astype(bool)
        & ~scenarios["baseline_peak_alert"].astype(bool)
    ]
    assert not new_alert_rows.empty
    scenario_name = str(new_alert_rows.iloc[0]["scenario"])
    scenario_data = repo.scenario_slice(origin, scenario_name)
    scenario_figure = charts.scenario_forecast_lines(scenario_data)
    assert any(trace.name == "New alert vs Baseline" for trace in scenario_figure.data)
    demand_comparison = charts.baseline_scenario_demand_lines(scenario_data, repo.fsas[0])
    risk_comparison = charts.baseline_scenario_risk_lines(scenario_data, repo.fsas[0], repo.threshold)
    assert demand_comparison.data[1].line.color == risk_comparison.data[1].line.color

    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "⚡ Ontario Electricity Demand & Peak-Risk Decision Support"
    assert any("page-section-title\">Decision Overview" in item.value for item in app.markdown)
    assert [metric.label for metric in app.metric] == [
        "Maximum combined selected-FSA demand",
        "Maximum FSA-level demand",
        "Unique alert hours",
        "FSA-hour alerts",
        "Max Demand FSA",
        "Highest-risk FSA",
    ]
    assert app.sidebar.caption[-1].value == (
        "Using this application requires recent demand history and a weather forecast "
        "to be preloaded and validated before model execution."
    )

    app.multiselect[0].set_value([repo.fsas[0]])
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == app.metric[1].value

    print("Redesigned Streamlit application: PASS")
    print("Default-page rendering: PASS")
    print("Run Analysis FSA update: PASS")
    print("Decision charts: PASS")
    print("Demand heatmap and alert markers: PASS")
    print("Weather-scenario alert markers and colors: PASS")


if __name__ == "__main__":
    main()
